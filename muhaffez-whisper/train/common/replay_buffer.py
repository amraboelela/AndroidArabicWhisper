"""
Replay buffer collection for preventing catastrophic forgetting

This module contains functions to collect samples from previously trained data
to include during training of new data, preventing the model from forgetting
what it learned earlier.
"""
import glob
import os
import random
import torch


def collect_replay_samples(dataset_name, current_surah_part, datasets_dir, current_set_size):
    """
    Collect a small sample from all previously trained surahs to prevent catastrophic forgetting.

    Args:
        dataset_name: Name of dataset (e.g., "Quran-A")
        current_surah_part: Current surah being trained (e.g., "002-01")
        datasets_dir: Path to datasets directory
        current_set_size: Size of current training set (to calculate 10% replay buffer with minimum 30)

    Returns:
        (replay_segment_files, replay_transcriptions): Lists of replay samples
    """
    current_surah_num = current_surah_part.split('-')[0]

    replay_segment_files = []
    replay_transcriptions = []

    # Find all text files for previous surah parts
    text_dir = f"../datasets/{dataset_name}/text"
    all_text_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))

    # Count previous surah parts and total available samples
    previous_surah_parts = []
    total_previous_samples = 0
    for text_file in all_text_files:
        basename = os.path.basename(text_file)
        surah_part = basename.replace('.txt', '')

        # Only include parts that are < current_surah_part
        if surah_part < current_surah_part:
            with open(text_file, "r", encoding="utf-8") as f:
                num_samples = len([line for line in f if line.strip()])
            previous_surah_parts.append((text_file, surah_part))
            total_previous_samples += num_samples

    if not previous_surah_parts:
        return replay_segment_files, replay_transcriptions

    # Calculate total replay buffer size
    total_replay_size = min(max(int(current_set_size * 0.1), 30), total_previous_samples)

    # Distribute replay budget evenly across previous surahs
    samples_per_surah = max(1, total_replay_size // len(previous_surah_parts))

    for text_file, surah_part in previous_surah_parts:
        # Load transcriptions
        with open(text_file, "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        # Extract surah number for mels path
        surah_num = surah_part.split('-')[0]

        # Load corresponding mel files
        if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
            # Multi-part surah (e.g., "002-04")
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels/normal", surah_num, surah_part, f"{surah_part}-*.pt")))
        else:
            # Single surah (e.g., "001")
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels/normal", surah_num, f"{surah_part}-*.pt")))

        # Fallback: try subdirectory if not found
        if not mel_files:
            mel_files = sorted(glob.glob(os.path.join(f"../datasets/{dataset_name}/mels/normal", surah_num, surah_part, f"{surah_part}-*.pt")))

        if len(mel_files) > 0 and len(mel_files) == len(transcriptions):
            num_samples = min(samples_per_surah, len(mel_files))
            indices = random.sample(range(len(mel_files)), num_samples)

            for idx in indices:
                replay_segment_files.append(mel_files[idx])
                replay_transcriptions.append(transcriptions[idx])

    if len(replay_segment_files) > 0:
        segment_names = [os.path.basename(f).replace('.pt', '') for f in replay_segment_files]
        segments_str = ', '.join(segment_names)
        print(f"  Replay buffer segments: ({segments_str})")
        print(f"  Replay buffer size: {len(replay_segment_files)}\n")

    return replay_segment_files, replay_transcriptions


def collect_curriculum_replay_samples(dataset_name, current_set_size):
    """
    Collect partial/chunked curriculum samples as replay buffer.
    Prevents catastrophic forgetting of curriculum patterns while training on full-length data.

    Args:
        dataset_name: Name of dataset (e.g., "Quran-A")
        current_set_size: Size of current training set (to calculate 10% replay buffer)

    Returns:
        List of tuples: [(file, transcription, target_seconds, target_words), ...]
    """
    CHUNK_DURATION = 1.3  # seconds per chunk
    WORDS_PER_CHUNK = 1   # words per chunk

    curriculum_replay_samples = []

    text_dir = f"../datasets/{dataset_name}/text"
    all_text_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))

    relevant_surah_parts = []
    for text_file in all_text_files:
        basename = os.path.basename(text_file)
        surah_part = basename.replace('.txt', '')
        relevant_surah_parts.append((text_file, surah_part))

    if not relevant_surah_parts:
        return curriculum_replay_samples

    # Calculate replay buffer size as 10% of current set
    replay_buffer_size = max(int(current_set_size * 0.1), 10)

    # Distribute evenly across surah parts
    samples_per_surah = max(1, replay_buffer_size // len(relevant_surah_parts))

    for text_file, surah_part in relevant_surah_parts:
        with open(text_file, "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        surah_num = surah_part.split('-')[0]

        # Load mel files
        if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
            mel_files = sorted(glob.glob(f"../datasets/{dataset_name}/mels/normal/{surah_num}/{surah_part}/{surah_part}-*.pt"))
        else:
            mel_files = sorted(glob.glob(f"../datasets/{dataset_name}/mels/normal/{surah_num}/{surah_part}-*.pt"))

        if not mel_files:
            mel_files = sorted(glob.glob(f"../datasets/{dataset_name}/mels/normal/{surah_num}/{surah_part}/{surah_part}-*.pt"))

        if len(mel_files) > 0 and len(mel_files) == len(transcriptions):
            num_samples = min(samples_per_surah, len(mel_files))
            indices = random.sample(range(len(mel_files)), num_samples)

            for idx in indices:
                mel_file = mel_files[idx]
                text = transcriptions[idx]

                # Get audio duration and calculate curriculum stages
                mel_features = torch.load(mel_file, map_location='cpu', weights_only=True)
                audio_duration = mel_features.shape[0] / 100.0
                num_words = len(text.split())

                num_chunks = int(audio_duration / CHUNK_DURATION)
                max_chunks = min(num_chunks, num_words)

                # Create curriculum samples at different difficulty levels
                for chunk_count in range(1, max_chunks + 1):
                    target_seconds = chunk_count * CHUNK_DURATION
                    target_words = chunk_count * WORDS_PER_CHUNK

                    curriculum_replay_samples.append((
                        mel_file,
                        text,
                        target_seconds,
                        target_words
                    ))

    # Shuffle and limit to replay_buffer_size
    random.shuffle(curriculum_replay_samples)
    curriculum_replay_samples = curriculum_replay_samples[:replay_buffer_size]

    if len(curriculum_replay_samples) > 0:
        print(f"  Curriculum replay buffer size: {len(curriculum_replay_samples)} partial segments\n")

    return curriculum_replay_samples


def collect_augmented_replay_samples(dataset_name, curriculum_set_size):
    """
    Collect replay samples from augmented data (which includes normal samples)
    to mix with curriculum training.

    Args:
        dataset_name: Name of dataset (e.g., "Quran-A")
        curriculum_set_size: Size of curriculum training set (to calculate 10% replay buffer)

    Returns:
        List of tuples: [(file, transcription, target_seconds, target_words), ...]
        where target_seconds and target_words are None for full-length samples
    """
    replay_samples = []

    # Calculate replay buffer size (10% of curriculum set)
    replay_size = max(int(curriculum_set_size * 0.1), 10)

    text_dir = f"../datasets/{dataset_name}/text"
    all_text_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))

    if not all_text_files:
        return replay_samples

    # Collect augmented samples
    augmented_samples = []
    augmentation_types = ['speed', 'pitch']

    for aug_type in augmentation_types:
        for text_file in all_text_files:
            basename = os.path.basename(text_file)
            surah_part = basename.replace('.txt', '')

            with open(text_file, "r", encoding="utf-8") as f:
                transcriptions = [line.strip() for line in f if line.strip()]

            surah_num = surah_part.split('-')[0]

            # Load augmented mel files
            aug_pattern_1 = f"../datasets/{dataset_name}/mels/augmented/{aug_type}/*/{surah_num}/{surah_part}-*.pt"
            aug_pattern_2 = f"../datasets/{dataset_name}/mels/augmented/{aug_type}/*/{surah_num}/{surah_part}/{surah_part}-*.pt"

            mel_files = sorted(glob.glob(aug_pattern_1))
            if not mel_files:
                mel_files = sorted(glob.glob(aug_pattern_2))

            # Match mel files with transcriptions
            # Augmented files have format: 002-01-001_speed_0.9.pt, etc.
            for mel_file in mel_files:
                mel_basename = os.path.basename(mel_file).replace('.pt', '')
                # Extract base segment name (e.g., "002-01-001" from "002-01-001_speed_0.9")
                base_name = mel_basename.split('_')[0]  # "002-01-001"

                # Find matching transcription by segment index
                seg_parts = base_name.split('-')
                if len(seg_parts) >= 3:
                    seg_idx = int(seg_parts[-1]) - 1  # Convert 1-based to 0-based index
                    if 0 <= seg_idx < len(transcriptions):
                        augmented_samples.append((mel_file, transcriptions[seg_idx], None, None))

    # Sample replay_size from augmented samples
    if len(augmented_samples) > 0:
        num_samples = min(replay_size, len(augmented_samples))
        replay_samples = random.sample(augmented_samples, num_samples)
        print(f"  Augmented replay buffer: {len(replay_samples)} samples (10%)\n")

    return replay_samples
