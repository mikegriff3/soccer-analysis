from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from utils.video_utils import read_video, save_video
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    input_video_path = '/Users/michaelgriffin/Football-Analysis/input_videos/08fd33_4.mp4'
    output_video_path = '/Users/michaelgriffin/Football-Analysis/output_videos/output_video1.mp4'
    
    # Read Video
    video_frames = read_video(input_video_path)
    
    # Initialize Tracker
    tracker = Tracker('./models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True, 
                                       stub_path='./stubs/track_stubs.pkl')
    
    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Save cropped image of a player
    # for track_id, player in tracks["players"][0].items():
    #     bbox = player["bbox"]
    #     frame = video_frames[0]

    #     # Crop bounding box from frame [(y1:y2), (x1:x2)] 
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     # Save the cropped image
    #     cv2.imwrite('./output_videos/cropped_image.jpg', cropped_image)
        
    #     break

    # Camera Movement Estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='./stubs/camera_movement_stub.pkl')

    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transform_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and Distance Estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)


    # Assign Players to Teams
    team_assigner = TeamAssigner()
    # Get the color of each team
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0]) 
    # Assign each player to a team
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)

            tracks['players'][frame_num][player_id]["team"] = team
            tracks['players'][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]


    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw Output
    ## Draw Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control) 

    # Draw Camera Movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw Speed and Distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    #Save Video
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()