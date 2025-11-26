"""Data collection utilities for training scripts"""
import glob
import os


def collect_augmented_data(dataset_name, text_files):
    """
    Collect regular and augmented segments for augmented training

    Returns:
        Tuple of (regular_segments, regular_transcriptions, all_training_segments, all_training_transcriptions)
    """
    regular_segment_files = []
    regular_transcriptions = []
    all_training_segments = []
    all_training_transcriptions = []

    datasets_dir = f"../datasets/{dataset_name}"

    for text_file in text_files:
        surah_part = os.path.splitext(os.path.basename(text_file))[0]
        surah_num = surah_part.split('-')[0]
        mels_dir = f"{datasets_dir}/mels/normal/{surah_num}"
        mels_augmented_dir = f"{datasets_dir}/mels/augmented"

        with open(text_file, "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        # Find regular mel files
        if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
            mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}/{surah_part}-*.pt"))
        else:
            mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}-*.pt"))

        if not mel_files:
            mel_files = sorted(glob.glob(f"{mels_dir}/{surah_part}/{surah_part}-*.pt"))

        if len(transcriptions) != len(mel_files):
            print(f"⚠️  Warning: Mismatch in {surah_part}")
            continue

        regular_segment_files.extend(mel_files)
        regular_transcriptions.extend(transcriptions)
        all_training_segments.extend(mel_files)
        all_training_transcriptions.extend(transcriptions)

        print(f"  Loaded {len(mel_files)} regular segments from {surah_part}")

        # Find augmented mel files
        augmented_variations = [
            'pitch/minus4', 'pitch/minus2', 'pitch/plus2', 'pitch/plus4',
            'speed/minus20', 'speed/minus10', 'speed/plus10', 'speed/plus20'
        ]

        augmented_count = 0
        has_augmented_data = False
        for aug_type in augmented_variations:
            if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
                aug_mel_files = sorted(glob.glob(f"{mels_augmented_dir}/{aug_type}/{surah_num}/{surah_part}/{surah_part}-*.pt"))
            else:
                aug_mel_files = sorted(glob.glob(f"{mels_augmented_dir}/{aug_type}/{surah_num}/{surah_part}-*.pt"))

            if aug_mel_files:
                all_training_segments.extend(aug_mel_files)
                all_training_transcriptions.extend(transcriptions)
                augmented_count += len(aug_mel_files)
                has_augmented_data = True

        if augmented_count > 0:
            print(f"  Loaded {augmented_count} augmented segments from {surah_part}")

        if not has_augmented_data:
            # Remove regular segments if no augmented data
            regular_segment_files = regular_segment_files[:-len(mel_files)]
            regular_transcriptions = regular_transcriptions[:-len(transcriptions)]
            all_training_segments = all_training_segments[:-len(mel_files)]
            all_training_transcriptions = all_training_transcriptions[:-len(transcriptions)]
            print(f"  Skipping {surah_part} - no augmented data available")

    return regular_segment_files, regular_transcriptions, all_training_segments, all_training_transcriptions


def collect_segment_files(dataset_name, text_files):
    """
    Collect segment files and transcriptions for regular training

    Returns:
        Tuple of (segment_files, transcriptions)
    """
    all_segment_files = []
    all_transcriptions = []
    datasets_dir = f"../datasets/{dataset_name}"

    for text_file in text_files:
        surah_part = os.path.basename(text_file).replace('.txt', '')
        surah_num = surah_part.split('-')[0]

        with open(text_file, 'r', encoding='utf-8') as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        # Find mel files
        if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
            segment_files = sorted(glob.glob(f"{datasets_dir}/mels/normal/{surah_num}/{surah_part}/{surah_part}-*.pt"))
        else:
            segment_files = sorted(glob.glob(f"{datasets_dir}/mels/normal/{surah_num}/{surah_part}-*.pt"))

        if not segment_files:
            print(f"⚠️  Skipping {surah_part}: no mel files found")
            continue

        all_segment_files.extend(segment_files)
        all_transcriptions.extend(transcriptions)

    return all_segment_files, all_transcriptions


def load_single_part_data(dataset_name, surah_part):
    """
    Load mel files and transcriptions for a single surah part

    Returns:
        Tuple of (segment_files, transcriptions)
    """
    datasets_dir = f"../datasets/{dataset_name}"
    mels_dir = f"{datasets_dir}/mels/normal"
    surah_num = surah_part.split('-')[0]

    text_path = f"{datasets_dir}/text/{surah_part}.txt"
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")

    with open(text_path, "r", encoding="utf-8") as f:
        transcriptions = [line.strip() for line in f if line.strip()]

    # Determine mel directory based on segment structure
    if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, surah_part, f"{surah_part}-*.pt")))
    else:
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, f"{surah_part}-*.pt")))

    if not segment_files:
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, surah_part, f"{surah_part}-*.pt")))
        if not segment_files:
            raise FileNotFoundError(f"No mel files found for {surah_part}")

    return segment_files, transcriptions
