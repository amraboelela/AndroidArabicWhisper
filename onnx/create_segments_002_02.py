#!/usr/bin/env python3
"""
Create segments for 002-02.mp3 based on text timing
Start: اول كافر به ولا تشتروا باياتي ثمنا قليلا واياي فاتقون
End: افتطمعون ان يؤمنوا لكم وقد كان فريق منهم يسمعون كلام الله ثم يحرفونه من بعد ما عقلوه وهم يعلمون
"""
import os
import subprocess

# Audio file info
audio_path = "/Users/amraboelela/audio/Quran-A/002-02.mp3"
total_duration = 900.0  # 15 minutes in seconds
segments_dir = "segments"

# Full text for 002-02 (ayat 65-76 approximately)
# Based on the pattern from 002-01, each line is one ayah or part of ayah
verses_text = [
    "اول كافر به ولا تشتروا باياتي ثمنا قليلا واياي فاتقون",
    "ولا تلبسوا الحق بالباطل وتكتموا الحق وانتم تعلمون",
    "واقيموا الصلاة واتوا الزكاة واركعوا مع الراكعين",
    "اتامرون الناس بالبر وتنسون انفسكم وانتم تتلون الكتاب",
    "افلا تعقلون",
    "واستعينوا بالصبر والصلاة وانها لكبيرة الا على الخاشعين",
    "الذين يظنون انهم ملاقوا ربهم وانهم اليه راجعون",
    "يا بني اسرائيل اذكروا نعمتي التي انعمت عليكم واني فضلتكم على العالمين",
    "واتقوا يوما لا تجزي نفس عن نفس شيئا ولا يقبل منها شفاعة ولا يؤخذ منها عدل ولا هم ينصرون",
    "واذ نجيناكم من ال فرعون يسومونكم سوء العذاب يذبحون ابناءكم ويستحيون نساءكم",
    "وفي ذلكم بلاء من ربكم عظيم",
    "واذ فرقنا بكم البحر فانجيناكم واغرقنا ال فرعون وانتم تنظرون",
    "واذ واعدنا موسى اربعين ليلة ثم اتخذتم العجل من بعده وانتم ظالمون",
    "ثم عفونا عنكم من بعد ذلك لعلكم تشكرون",
    "واذ اتينا موسى الكتاب والفرقان لعلكم تهتدون",
    "واذ قال موسى لقومه يا قوم انكم ظلمتم انفسكم باتخاذكم العجل فتوبوا الى بارئكم فاقتلوا انفسكم",
    "فاقتلوا انفسكم ذلكم خير لكم عند بارئكم فتاب عليكم",
    "انه هو التواب الرحيم",
    "واذ قلتم يا موسى لن نؤمن لك حتى نرى الله جهرة فاخذتكم الصاعقة وانتم تنظرون",
    "ثم بعثناكم من بعد موتكم لعلكم تشكرون",
    "وظللنا عليكم الغمام وانزلنا عليكم المن والسلوى",
    "كلوا من طيبات ما رزقناكم وما ظلمونا ولكن كانوا انفسهم يظلمون",
    "واذ قلنا ادخلوا هذه القرية فكلوا منها حيث شئتم رغدا وادخلوا الباب سجدا وقولوا حطة نغفر لكم خطاياكم",
    "وسنزيد المحسنين",
    "فبدل الذين ظلموا قولا غير الذي قيل لهم فانزلنا على الذين ظلموا رجزا من السماء بما كانوا يفسقون",
    "واذ استسقى موسى لقومه فقلنا اضرب بعصاك الحجر فانفجرت منه اثنتا عشرة عينا",
    "قد علم كل اناس مشربهم كلوا واشربوا من رزق الله ولا تعثوا في الارض مفسدين",
    "واذ قلتم يا موسى لن نصبر على طعام واحد فادع لنا ربك يخرج لنا مما تنبت الارض من بقلها وقثائها وفومها وعدسها وبصلها",
    "قال اتستبدلون الذي هو ادنى بالذي هو خير",
    "اهبطوا مصرا فان لكم ما سالتم وضربت عليهم الذلة والمسكنة وباءوا بغضب من الله",
    "ذلك بانهم كانوا يكفرون بايات الله ويقتلون النبيين بغير الحق ذلك بما عصوا وكانوا يعتدون",
    "ان الذين امنوا والذين هادوا والنصارى والصابئين من امن بالله واليوم الاخر وعمل صالحا فلهم اجرهم عند ربهم",
    "ولا خوف عليهم ولا هم يحزنون",
    "واذ اخذنا ميثاقكم ورفعنا فوقكم الطور خذوا ما اتيناكم بقوة واذكروا ما فيه لعلكم تتقون",
    "ثم توليتم من بعد ذلك فلولا فضل الله عليكم ورحمته لكنتم من الخاسرين",
    "ولقد علمتم الذين اعتدوا منكم في السبت فقلنا لهم كونوا قردة خاسئين",
    "فجعلناها نكالا لما بين يديها وما خلفها وموعظة للمتقين",
    "واذ قال موسى لقومه ان الله يامركم ان تذبحوا بقرة قالوا اتتخذنا هزوا قال اعوذ بالله ان اكون من الجاهلين",
    "قالوا ادع لنا ربك يبين لنا ما هي قال انه يقول انها بقرة لا فارض ولا بكر عوان بين ذلك فافعلوا ما تؤمرون",
    "قالوا ادع لنا ربك يبين لنا ما لونها قال انه يقول انها بقرة صفراء فاقع لونها تسر الناظرين",
    "قالوا ادع لنا ربك يبين لنا ما هي ان البقر تشابه علينا وانا ان شاء الله لمهتدون",
    "قال انه يقول انها بقرة لا ذلول تثير الارض ولا تسقي الحرث مسلمة لا شية فيها",
    "قالوا الان جئت بالحق فذبحوها وما كادوا يفعلون",
    "واذ قتلتم نفسا فادارءتم فيها والله مخرج ما كنتم تكتمون",
    "فقلنا اضربوه ببعضها كذلك يحيي الله الموتى ويريكم اياته لعلكم تعقلون",
    "ثم قست قلوبكم من بعد ذلك فهي كالحجارة او اشد قسوة",
    "وان من الحجارة لما يتفجر منه الانهار وان منها لما يشقق فيخرج منه الماء وان منها لما يهبط من خشية الله",
    "وما الله بغافل عما تعملون",
    "افتطمعون ان يؤمنوا لكم وقد كان فريق منهم يسمعون كلام الله ثم يحرفونه من بعد ما عقلوه وهم يعلمون",
]

# Count total words
total_words = sum(len(verse.split()) for verse in verses_text)
print(f"Total verses: {len(verses_text)}")
print(f"Total words: {total_words}")
print(f"Total duration: {total_duration:.1f} seconds")
print(f"Average: {total_duration / total_words:.2f} seconds per word")

# Create segments based on word count
segments = []
current_time = 0.0
seconds_per_word = total_duration / total_words

for i, verse_text in enumerate(verses_text, 1):
    words = verse_text.split()
    num_words = len(words)

    # Calculate segment duration based on word count
    segment_duration = num_words * seconds_per_word

    # Extract segment using ffmpeg
    segment_filename = f"{segments_dir}/002-02-{i:03d}.wav"

    # Make sure we don't exceed audio length
    if current_time + segment_duration > total_duration:
        segment_duration = total_duration - current_time

    # ffmpeg command to extract segment
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(current_time),
        "-t", str(segment_duration),
        "-i", audio_path,
        "-ar", "16000",  # 16kHz sample rate for Whisper
        "-ac", "1",       # mono
        segment_filename
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    segments.append({
        "index": i,
        "text": verse_text,
        "words": num_words,
        "start": current_time,
        "end": current_time + segment_duration,
        "duration": segment_duration,
        "filename": segment_filename
    })

    print(f"Segment {i:03d}: {segment_duration:.1f}s ({num_words} words) - {verse_text[:50]}...")

    current_time += segment_duration

# Save text file
text_filename = f"{segments_dir}/002-02.txt"
with open(text_filename, "w", encoding="utf-8") as f:
    for segment in segments:
        f.write(segment["text"] + "\n")

print(f"\n✓ Created {len(segments)} segments")
print(f"✓ Saved text to: {text_filename}")
print(f"✓ Total duration used: {current_time:.1f}s / {total_duration:.1f}s")
