#!/usr/bin/env python3
"""
Create text file for 002-02 segments by mapping Quran verses to segment durations
"""
import os
import glob
import torchaudio

# Quran text for 002-02
# Start: اول كافر به ولا تشتروا باياتي ثمنا قليلا واياي فاتقون
# End: افتطمعون ان يؤمنوا لكم وقد كان فريق منهم يسمعون كلام الله ثم يحرفونه من بعد ما عقلوه وهم يعلمون
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

datasets_dir = "datasets/base"

# Get all 002-02 segment files and their durations
segment_files = sorted(glob.glob(os.path.join(datasets_dir, "002-02-*.wav")))
print(f"Found {len(segment_files)} audio segments")

# Calculate durations and word counts
segment_durations = []
for seg_file in segment_files:
    waveform, sample_rate = torchaudio.load(seg_file)
    duration = waveform.shape[1] / sample_rate
    segment_durations.append(duration)

total_audio_duration = sum(segment_durations)
total_words = sum(len(verse.split()) for verse in verses_text)

print(f"Total verses: {len(verses_text)}")
print(f"Total words in text: {total_words}")
print(f"Total audio duration: {total_audio_duration:.1f}s")
print(f"Average: {total_audio_duration / total_words:.2f} seconds per word")

# Map verses to segments based on duration
seconds_per_word = total_audio_duration / total_words
segment_texts = []
verse_index = 0
current_verse_words = verses_text[verse_index].split()
current_verse_word_index = 0

for i, duration in enumerate(segment_durations):
    # How many words fit in this segment?
    words_in_segment = max(1, round(duration / seconds_per_word))

    segment_words = []
    words_collected = 0

    while words_collected < words_in_segment and verse_index < len(verses_text):
        words_remaining_in_verse = len(current_verse_words) - current_verse_word_index
        words_needed = words_in_segment - words_collected

        if words_needed >= words_remaining_in_verse:
            # Take rest of current verse
            segment_words.extend(current_verse_words[current_verse_word_index:])
            words_collected += words_remaining_in_verse

            # Move to next verse
            verse_index += 1
            if verse_index < len(verses_text):
                current_verse_words = verses_text[verse_index].split()
                current_verse_word_index = 0
        else:
            # Take partial verse
            segment_words.extend(current_verse_words[current_verse_word_index:current_verse_word_index + words_needed])
            current_verse_word_index += words_needed
            words_collected += words_needed

    segment_text = " ".join(segment_words)
    segment_texts.append(segment_text)
    print(f"Segment {i+1:3d}: {duration:5.2f}s ({len(segment_words):2d} words) - {segment_text[:50]}...")

# Save to text file
text_filename = os.path.join(datasets_dir, "002-02.txt")
with open(text_filename, "w", encoding="utf-8") as f:
    for text in segment_texts:
        f.write(text + "\n")

print(f"\n✓ Created {len(segment_texts)} text entries")
print(f"✓ Saved to: {text_filename}")
