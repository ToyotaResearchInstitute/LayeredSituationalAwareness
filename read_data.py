import pandas as pd
import numpy as np
import os
import argparse
import cv2
from PIL import Image
import io
from skimage.filters import threshold_otsu
from skimage import color


def process_raw_gaze(gaze_data):
    processed_gaze_data = {
        'timestamp': [],
        'x': [],
        'y': [],
        'video_timestamp': []
    }

    for _, row in gaze_data.iterrows():
        lx = row['lx']
        ly = row['ly']
        rx = row['rx']
        ry = row['ry']
        # if both are nan, skip; if one is nan, use the other; otherwise average
        if np.isnan(lx) and np.isnan(rx):
            continue
        elif np.isnan(lx):
            x = rx
            y = ry
        elif np.isnan(rx):
            x = lx
            y = ly
        else:
            x = (lx + rx) / 2.0
            y = (ly + ry) / 2.0

        processed_gaze_data['timestamp'].append(row['video_ts'])
        processed_gaze_data['x'].append(x)
        processed_gaze_data['y'].append(y)
        processed_gaze_data['video_timestamp'].append(row['video_ts'])

    processed_gaze_data = pd.DataFrame(processed_gaze_data)
    processed_gaze_data['confidence'] = np.array([1.0] * len(processed_gaze_data))

    return processed_gaze_data


def extract_continuous_sequences(nums):
    nums = sorted(nums)
    sequences = [[nums[0]]]

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            sequences[-1].append(nums[i])
        else:
            sequences.append([nums[i]])

    return sequences


def get_contrast_value(image, center_x, center_y, bb_w, bb_h):
    x1 = center_x - bb_w // 2
    x2 = center_x + bb_w // 2
    y1 = center_y - bb_h // 2
    y2 = center_y + bb_h // 2

    bbox_region = image[y1:y2, x1:x2]

    H, W = image.shape[:2]
    mask = np.ones((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 0

    bbox_lab = color.rgb2lab(bbox_region)
    rest_lab = color.rgb2lab(image[mask.astype(bool)].reshape(-1, 3))

    mean_bbox_lab = np.mean(bbox_lab.reshape(-1, 3), axis=0)
    mean_rest_lab = np.mean(rest_lab, axis=0)

    # Delta E (CIE76)
    delta_e = np.linalg.norm(mean_bbox_lab - mean_rest_lab)
    return delta_e


def main():
    parser = argparse.ArgumentParser(description='Process raw gaze data and extract features.')
    parser.add_argument('--raw_data_path', type=str, default='raw_data',
                        help='Path to raw data directory (containing transcriptions.csv, gaze_data/, queries.csv)')
    parser.add_argument('--processed_data_path', type=str, default='processed_data',
                        help='Output directory for processed feature CSVs')
    parser.add_argument('--waymo_camera_path', type=str, required=True,
                        help='Path to Waymo camera_image parquet directory')
    parser.add_argument('--waymo_bb_path', type=str, required=True,
                        help='Path to Waymo camera_box parquet directory')
    args = parser.parse_args()

    os.makedirs(args.processed_data_path, exist_ok=True)

    transcriptions = pd.read_csv(os.path.join(args.raw_data_path, 'transcriptions.csv'))
    qf = pd.read_csv(os.path.join(args.raw_data_path, 'queries.csv'))
    contrast_dict = {}

    for samp in range(len(transcriptions)):
        user = transcriptions['subject_id'][samp]
        gaze_file_path = os.path.join(args.raw_data_path, 'gaze_data', transcriptions['gaze_file_path'][samp])
        gaze_file = pd.read_csv(gaze_file_path)
        gaze_data = process_raw_gaze(gaze_file)

        points = np.vstack([gaze_data['x'].tolist(), gaze_data['y'].tolist()]).T
        velocities = np.sqrt(((points[:-1] - points[1:])**2).sum(1))

        try:
            boundary = threshold_otsu(velocities)
        except Exception as e:
            print(e)
            print("**********************************************", user, samp)
            continue

        fixations = extract_continuous_sequences(np.argwhere(velocities < boundary).flatten())
        vid_name = transcriptions['video_file'][samp]
        df_name = vid_name.replace('.mp4', '.parquet')
        camera_df = pd.read_parquet(os.path.join(args.waymo_camera_path, df_name))
        camera_df = camera_df[camera_df['key.camera_name'] == 1]
        camera_times = np.array(camera_df['key.frame_timestamp_micros'].tolist())
        bb_df = pd.read_parquet(os.path.join(args.waymo_bb_path, df_name))
        bb_df = bb_df[bb_df['key.camera_name'] == 1]

        t0 = camera_df['key.frame_timestamp_micros'].iloc[0]
        gaze_unnorm_x = 1920
        gaze_unnorm_y = 1280

        all_fixs = [item for sublist in fixations if len(sublist) > 5 for item in sublist]
        time = gaze_data.iloc[all_fixs]['video_timestamp']
        tf = t0 + time * 1e3  # Convert milliseconds to microseconds
        tf = np.array(tf)
        diffs = np.abs(camera_times - tf[:, np.newaxis])
        closest_time_idx = np.argmin(diffs, axis=1)
        sel_gaze_data = gaze_data.iloc[all_fixs].copy()
        sel_gaze_data['key.frame_timestamp_micros'] = camera_times[closest_time_idx]

        question_image = int(os.path.basename(transcriptions.iloc[samp]['question_image']).split("_")[-2])
        object_id = qf['object_id'].iloc[question_image]

        req_cam_df = sel_gaze_data.merge(camera_df, on='key.frame_timestamp_micros', how='inner')

        bb_df = bb_df[bb_df['key.camera_object_id'] == object_id]
        overall_df = pd.merge(req_cam_df, bb_df, on='key.frame_timestamp_micros', how='left')
        gaze_dist_obj = ((overall_df['x'] * gaze_unnorm_x - overall_df['[CameraBoxComponent].box.center.x'])**2 + (overall_df['y'] * gaze_unnorm_y - overall_df['[CameraBoxComponent].box.center.y'])**2)**0.5
        overall_df['gaze_dist_obj'] = gaze_dist_obj

        g_dist_pause = overall_df['gaze_dist_obj'].iloc[-1]
        g_dist_avg = overall_df[overall_df['gaze_dist_obj'].notna()]['gaze_dist_obj'].mean()
        g_dist_min = overall_df[overall_df['gaze_dist_obj'].notna()]['gaze_dist_obj'].min()

        comp_gaze_data = {
            'q_image': [],
            'gaze_dist_pause': [],
            'gaze_dist_avg': [],
            'gaze_dist_min': [],
            'contrasts': [],
            'object_size': [],
            'object_distance': [],
            'object_duration': []
        }

        for i in range(4):
            comp_gaze_data[f'h_avg_{i+1}'] = []
            comp_gaze_data[f'h_dwell_time_{i+1}'] = []
            comp_gaze_data[f'h_elapse_time_{i+1}'] = []

        comp_gaze_data['gaze_dist_pause'].append(g_dist_pause)
        comp_gaze_data['gaze_dist_avg'].append(g_dist_avg)
        comp_gaze_data['gaze_dist_min'].append(g_dist_min)

        viewing_distance_mm = 680
        pixels_per_mm = 4.3
        radius_px_1 = int(viewing_distance_mm * np.tan(np.deg2rad(2.5)) * pixels_per_mm)
        radius_px_2 = int(viewing_distance_mm * np.tan(np.deg2rad(4.1)) * pixels_per_mm)
        radius_px_3 = int(viewing_distance_mm * np.tan(np.deg2rad(9.1)) * pixels_per_mm)
        radius_px_4 = int(viewing_distance_mm * np.tan(np.deg2rad(15.0)) * pixels_per_mm)
        rads = np.array([radius_px_1, radius_px_2, radius_px_3, radius_px_4])
        g_dist_arr = overall_df['gaze_dist_obj'].values
        dwell_times = (g_dist_arr[:, np.newaxis] < rads)
        h_elapses = []
        h_avgs = []
        h_dwells = []
        for i in range(4):
            inds = np.where(dwell_times[:, i] == True)
            max_ind = inds[-1][-1] if len(inds[0]) > 0 else 0
            h_elapse = overall_df['key.frame_timestamp_micros'].iloc[-1] - overall_df['key.frame_timestamp_micros'].iloc[max_ind]
            h_elapses.append(h_elapse)
            h_dwell = len(inds[0])
            h_dwells.append(h_dwell)
            h_avg = g_dist_arr[inds[0]]
            if len(h_avg) > 0:
                h_avg = np.mean(h_avg)
            else:
                h_avg = np.nan
            h_avgs.append(h_avg)

        for i in range(4):
            comp_gaze_data[f'h_avg_{i+1}'].append(h_avgs[i])
            comp_gaze_data[f'h_dwell_time_{i+1}'].append(h_dwells[i])
            comp_gaze_data[f'h_elapse_time_{i+1}'].append(h_elapses[i])

        if transcriptions.iloc[samp]['question_image'] in contrast_dict:
            contrast = contrast_dict[transcriptions.iloc[samp]['question_image']]
            comp_gaze_data['q_image'].append(transcriptions.iloc[samp]['question_image'])
            comp_gaze_data['contrasts'].append(contrast)
        else:
            question_time = transcriptions.iloc[samp]['frame_end'] * 100
            t0 = camera_df['key.frame_timestamp_micros'].iloc[0]
            tf = t0 + question_time * 1e3
            camera_time_rows = camera_df.loc[camera_df[(camera_df['key.frame_timestamp_micros'] >= tf)]['key.frame_timestamp_micros'].idxmin()]

            time_bb_row = bb_df[bb_df['key.frame_timestamp_micros'] == camera_time_rows['key.frame_timestamp_micros']]
            time_bb_row = time_bb_row[time_bb_row['key.camera_object_id'] == object_id]
            x = time_bb_row['[CameraBoxComponent].box.center.x'].iloc[0]
            y = time_bb_row['[CameraBoxComponent].box.center.y'].iloc[0]
            w = time_bb_row['[CameraBoxComponent].box.size.x'].iloc[0]
            h = time_bb_row['[CameraBoxComponent].box.size.y'].iloc[0]

            image = camera_time_rows['[CameraImageComponent].image']
            image = np.array(Image.open(io.BytesIO(image)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            contrast = get_contrast_value(image, int(x), int(y), int(w), int(h))
            comp_gaze_data['q_image'].append(transcriptions.iloc[samp]['question_image'])
            comp_gaze_data['contrasts'].append(contrast)
            contrast_dict[transcriptions.iloc[samp]['question_image']] = contrast

        object_size = overall_df.iloc[-1]['[CameraBoxComponent].box.size.x'] * overall_df.iloc[-1]['[CameraBoxComponent].box.size.y']
        object_pos_x = overall_df.iloc[-1]['[CameraBoxComponent].box.center.x']
        object_pos_y = overall_df.iloc[-1]['[CameraBoxComponent].box.center.y']
        object_distance = np.sqrt((object_pos_x - 960)**2 + (object_pos_y - 512)**2)
        object_duration = len(overall_df[overall_df['key.camera_object_id'].notna()]['key.frame_timestamp_micros'])
        comp_gaze_data['object_size'].append(object_size)
        comp_gaze_data['object_distance'].append(object_distance)
        comp_gaze_data['object_duration'].append(object_duration)

        comp_gaze_data = pd.DataFrame(comp_gaze_data)
        comp_gaze_data.to_csv(os.path.join(args.processed_data_path, f'{samp}.csv'), index=False)
        print(f"Processed sample {samp} for user {user}")


if __name__ == '__main__':
    main()
