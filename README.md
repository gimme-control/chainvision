# Gun Person Tracker

A single-camera system that detects people with guns and maintains their identity even when they leave and return to the camera view.

## Features

- **Gun Detection**: Uses custom YOLO model to detect guns
- **Person Detection**: Uses YOLOv8 to detect people
- **Gun-Person Matching**: Associates guns with the people holding them
- **Person Re-identification**: Maintains person identity using OSNet ReID model
- **Persistent Tracking**: Remembers people even when they leave camera view
- **Database Storage**: Saves person embeddings and metadata to disk

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the main tracking system:

```bash
python gun_person_tracker.py
```

### Controls

- **'q'**: Quit the application
- **'s'**: Save the person database to disk

## How It Works

1. **Detection**: The system detects both guns and people in each frame
2. **Matching**: Guns are matched to nearby people using IoU (Intersection over Union)
3. **ReID**: Each person gets an embedding using OSNet model
4. **Tracking**: People are tracked across frames using cosine similarity
5. **Persistence**: Person identities are maintained even when they leave view
6. **Database**: All person data is saved to `person_database.json`

## Configuration

Edit `config.py` to adjust:
- Model paths
- Detection thresholds
- ReID parameters
- Camera settings

## File Structure

- `gun_person_tracker.py`: Main tracking system
- `config.py`: Configuration settings
- `person_database.json`: Saved person database (created automatically)
- `requirements.txt`: Python dependencies

## Technical Details

- **Gun Model**: Custom YOLO model trained on gun detection
- **Person Model**: YOLOv8n for person detection
- **ReID Model**: OSNet for person re-identification
- **Similarity**: Cosine similarity with temporal consistency
- **Database**: JSON-based storage with automatic loading/saving