#!/usr/bin/env python3
"""
Create text file for 002-02 segments by keeping one ayah per line
Split longest ayat to reach exactly 63 lines
"""
import os

# Quran text for 002-02 (ayat 41-75)
# Start: اول كافر به ولا تشتروا باياتي ثمنا قليلا
# End: افتطمعون ان يؤمنوا لكم وقد كان فريق منهم يسمعون كلام الله ثم يحرفونه من بعد ما عقلوه وهم يعلمون
verses_text = [
    "اول كافر به ولا تشتروا باياتي ثمنا قليلا واياي فاتقون",
    "ولا تلبسوا الحق بالباطل وتكتموا الحق وانتم تعلمون",
    "واقيموا الصلاة واتوا الزكاة واركعوا مع الراكعين",
    "اتامرون الناس بالبر وتنسون انفسكم وانتم تتلون الكتاب افلا تعقلون",
    "واستعينوا بالصبر والصلاة وانها لكبيرة الا على الخاشعين",
    "الذين يظنون انهم ملاقوا ربهم وانهم اليه راجعون",
    "يا بني اسرائيل اذكروا نعمتي التي انعمت عليكم واني فضلتكم على العالمين",
    "واتقوا يوما لا تجزي نفس عن نفس شيئا ولا يقبل منها شفاعة ولا يؤخذ منها عدل ولا هم ينصرون",
    "واذ نجيناكم من ال فرعون يسومونكم سوء العذاب يذبحون ابناءكم ويستحيون نساءكم وفي ذلكم بلاء من ربكم عظيم",
    "واذ فرقنا بكم البحر فانجيناكم واغرقنا ال فرعون وانتم تنظرون",
    "واذ واعدنا موسى اربعين ليلة ثم اتخذتم العجل من بعده وانتم ظالمون",
    "ثم عفونا عنكم من بعد ذلك لعلكم تشكرون",
    "واذ اتينا موسى الكتاب والفرقان لعلكم تهتدون",
    "واذ قال موسى لقومه يا قوم انكم ظلمتم انفسكم باتخاذكم العجل فتوبوا الى بارئكم فاقتلوا انفسكم ذلكم خير لكم عند بارئكم فتاب عليكم انه هو التواب الرحيم",
    "واذ قلتم يا موسى لن نؤمن لك حتى نرى الله جهرة فاخذتكم الصاعقة وانتم تنظرون",
    "ثم بعثناكم من بعد موتكم لعلكم تشكرون",
    "وظللنا عليكم الغمام وانزلنا عليكم المن والسلوى كلوا من طيبات ما رزقناكم وما ظلمونا ولكن كانوا انفسهم يظلمون",
    "واذ قلنا ادخلوا هذه القرية فكلوا منها حيث شئتم رغدا وادخلوا الباب سجدا وقولوا حطة نغفر لكم خطاياكم وسنزيد المحسنين",
    "فبدل الذين ظلموا قولا غير الذي قيل لهم فانزلنا على الذين ظلموا رجزا من السماء بما كانوا يفسقون",
    "واذ استسقى موسى لقومه فقلنا اضرب بعصاك الحجر فانفجرت منه اثنتا عشرة عينا قد علم كل اناس مشربهم كلوا واشربوا من رزق الله ولا تعثوا في الارض مفسدين",
    "واذ قلتم يا موسى لن نصبر على طعام واحد فادع لنا ربك يخرج لنا مما تنبت الارض من بقلها وقثائها وفومها وعدسها وبصلها قال اتستبدلون الذي هو ادنى بالذي هو خير اهبطوا مصرا فان لكم ما سالتم وضربت عليهم الذلة والمسكنة وباءوا بغضب من الله ذلك بانهم كانوا يكفرون بايات الله ويقتلون النبيين بغير الحق ذلك بما عصوا وكانوا يعتدون",
    "ان الذين امنوا والذين هادوا والنصارى والصابئين من امن بالله واليوم الاخر وعمل صالحا فلهم اجرهم عند ربهم ولا خوف عليهم ولا هم يحزنون",
    "واذ اخذنا ميثاقكم ورفعنا فوقكم الطور خذوا ما اتيناكم بقوة واذكروا ما فيه لعلكم تتقون",
    "ثم توليتم من بعد ذلك فلولا فضل الله عليكم ورحمته لكنتم من الخاسرين",
    "ولقد علمتم الذين اعتدوا منكم في السبت فقلنا لهم كونوا قردة خاسئين",
    "فجعلناها نكالا لما بين يديها وما خلفها وموعظة للمتقين",
    "واذ قال موسى لقومه ان الله يامركم ان تذبحوا بقرة قالوا اتتخذنا هزوا قال اعوذ بالله ان اكون من الجاهلين",
    "قالوا ادع لنا ربك يبين لنا ما هي قال انه يقول انها بقرة لا فارض ولا بكر عوان بين ذلك فافعلوا ما تؤمرون",
    "قالوا ادع لنا ربك يبين لنا ما لونها قال انه يقول انها بقرة صفراء فاقع لونها تسر الناظرين",
    "قالوا ادع لنا ربك يبين لنا ما هي ان البقر تشابه علينا وانا ان شاء الله لمهتدون",
    "قال انه يقول انها بقرة لا ذلول تثير الارض ولا تسقي الحرث مسلمة لا شية فيها قالوا الان جئت بالحق فذبحوها وما كادوا يفعلون",
    "واذ قتلتم نفسا فادارءتم فيها والله مخرج ما كنتم تكتمون",
    "فقلنا اضربوه ببعضها كذلك يحيي الله الموتى ويريكم اياته لعلكم تعقلون",
    "ثم قست قلوبكم من بعد ذلك فهي كالحجارة او اشد قسوة وان من الحجارة لما يتفجر منه الانهار وان منها لما يشقق فيخرج منه الماء وان منها لما يهبط من خشية الله وما الله بغافل عما تعملون",
    "افتطمعون ان يؤمنوا لكم وقد كان فريق منهم يسمعون كلام الله ثم يحرفونه من بعد ما عقلوه وهم يعلمون",
]

print(f"Total ayat: {len(verses_text)}")
print(f"Target lines: 63")
print(f"Need to split: {63 - len(verses_text)} ayat")

# Calculate word count for each ayah
ayat_with_lengths = []
for i, ayah in enumerate(verses_text):
    words = ayah.split()
    ayat_with_lengths.append({
        'index': i,
        'text': ayah,
        'word_count': len(words),
        'lines': [ayah]  # Start with whole ayah as one line
    })

# Sort by word count (longest first) to identify which ayat to split
sorted_ayat = sorted(ayat_with_lengths, key=lambda x: x['word_count'], reverse=True)

print("\nLongest ayat:")
for i in range(min(10, len(sorted_ayat))):
    ayah = sorted_ayat[i]
    print(f"  Ayah {ayah['index']+1}: {ayah['word_count']} words - {ayah['text'][:60]}...")

# Split longest ayat until we reach 63 lines
splits_needed = 63 - len(verses_text)
print(f"\nSplitting {splits_needed} longest ayat into 2 lines each...")

for i in range(splits_needed):
    ayah = sorted_ayat[i]
    words = ayah['text'].split()
    mid = len(words) // 2

    # Split into two lines
    line1 = ' '.join(words[:mid])
    line2 = ' '.join(words[mid:])

    ayah['lines'] = [line1, line2]
    print(f"  Split ayah {ayah['index']+1} ({ayah['word_count']} words):")
    print(f"    Line 1: {len(line1.split())} words - {line1[:50]}...")
    print(f"    Line 2: {len(line2.split())} words - {line2[:50]}...")

# Rebuild in original order
final_lines = []
for ayah in ayat_with_lengths:
    final_lines.extend(ayah['lines'])

print(f"\n✓ Total lines: {len(final_lines)}")

# Save to text file
segments_dir = "segments"
text_filename = os.path.join(segments_dir, "002-02.txt")
with open(text_filename, "w", encoding="utf-8") as f:
    for line in final_lines:
        f.write(line + "\n")

print(f"✓ Saved to: {text_filename}")

# Show statistics
total_words = sum(len(line.split()) for line in final_lines)
print(f"\nStatistics:")
print(f"  Total lines: {len(final_lines)}")
print(f"  Total words: {total_words}")
print(f"  Average words per line: {total_words / len(final_lines):.1f}")
