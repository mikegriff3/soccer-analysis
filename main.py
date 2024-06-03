from utils import read_video, save_video
from trackers import Tracker

from utils.video_utils import read_video, save_video

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
    
    # Draw Output
    ## Draw Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks) 

    #Save Video
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()