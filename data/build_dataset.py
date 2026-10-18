"""Builds the complaint dataset.

1. train.csv  - template-assisted complaints (issue phrases x openers x urgency
                suffixes) with random Roman Urdu spelling noise. Labels:
                category, sentiment, priority.
2. test_natural.csv - hand-written complaints with different wording than the
                templates, used as a harder held-out set.

Run:  python data/build_dataset.py
"""
import random
import pandas as pd
from pathlib import Path

random.seed(42)
OUT = Path(__file__).parent

# --------------------------------------------------------------------------
# Issue phrases per category: (negative, neutral/query, positive)
# --------------------------------------------------------------------------
ISSUES = {
    "LMS": {
        "neg": [
            "LMS pe assignment submit nahi ho rahi",
            "LMS login nahi ho raha password galat bata raha hai",
            "LMS pe quiz open nahi ho raha",
            "LMS mein lecture videos load nahi hotin",
            "LMS pe meri uploaded file show nahi ho rahi",
            "mera course LMS mein nazar nahi aa raha",
            "LMS ka server down hai",
            "LMS pe marks update nahi hue",
            "LMS pe assignment submit karte hue error aa raha hai",
            "LMS se course material download nahi ho raha",
            "my assignment is not submitting on LMS",
            "LMS quiz timer ghalat chal raha hai aur submit bhi nahi hua",
        ],
        "neu": [
            "LMS pe assignment kahan submit karna hai",
            "LMS ka password reset kaise hoga",
            "LMS pe quiz ka schedule kahan milega",
            "LMS mein course enroll kaise karte hain",
            "LMS pe lecture slides kab upload hongi",
            "how can I access my course on LMS",
        ],
        "pos": [
            "LMS ka naya interface bohat acha hai",
            "LMS pe assignment submit karna ab bohat asaan hai",
            "LMS ki lecture recordings bohat helpful hain",
        ],
    },
    "Fee": {
        "neg": [
            "fee challan generate nahi ho raha",
            "fee jama kar di lekin portal pe show nahi ho rahi",
            "fee challan mein galat amount aa raha hai",
            "late fee ghalat charge hui hai",
            "bank se fee payment ho gayi lekin confirm nahi hui",
            "scholarship ka amount fee se minus nahi hua",
            "challan par meri detail galat hai",
            "fee refund abhi tak nahi mila",
            "my fee challan is showing wrong amount",
            "fee do dafa deduct ho gayi hai",
        ],
        "neu": [
            "fee installment ka option hai kya",
            "fee jama karne ki last date kya hai",
            "fee challan kahan se download karun",
            "scholarship ke liye kaise apply karte hain",
            "fee kis bank mein jama hoti hai",
            "what is the fee structure for this semester",
        ],
        "pos": [
            "fee installment ki facility ke liye shukriya",
            "fee ka process bohat smooth tha",
            "scholarship milne par bohat shukriya",
        ],
    },
    "Registration": {
        "neg": [
            "course registration nahi ho rahi",
            "registration portal open nahi ho raha",
            "course add nahi ho raha section full bata raha hai",
            "registration mein meri semester galat show ho rahi hai",
            "course drop karne ka option nahi aa raha",
            "registration confirm hone ke baad bhi course list mein nahi hai",
            "enrollment form submit nahi ho raha",
            "meri registration bina wajah ke reject ho gayi",
            "I am unable to register for my courses",
            "section change request abhi tak approve nahi hui",
        ],
        "neu": [
            "course registration ki last date kab hai",
            "course add drop kaise karte hain",
            "registration ke liye kya documents chahiye",
            "semester freeze kaise karwate hain",
            "section change ho sakta hai kya",
            "how do I register for a pre-requisite course",
        ],
        "pos": [
            "is dafa registration process bohat asaan tha",
            "registration team ka shukriya jaldi kaam kar diya",
            "online registration ki facility bohat acha hai",
        ],
    },
    "Examination": {
        "neg": [
            "exam schedule abhi tak upload nahi hua",
            "roll number slip nahi mil rahi",
            "paper ke marks galat lage hain",
            "result mein meri grade galat hai",
            "exam ki date do papers ke saath clash kar rahi hai",
            "admit card download nahi ho raha",
            "re-checking ka result abhi tak nahi aaya",
            "paper mein out of syllabus questions aaye",
            "my exam result shows wrong grade",
            "midterm ka result abhi tak nahi aaya",
        ],
        "neu": [
            "midterm ka schedule kab aayega",
            "re-checking ke liye kaise apply karun",
            "exam hall ka seating plan kahan hai",
            "paper ka format kya hoga",
            "result kab announce hoga",
            "when will the makeup exam be held",
        ],
        "pos": [
            "exam ka paper bohat fair tha shukriya",
            "result jaldi announce karne ka shukriya",
            "exam arrangements bohat acha thay",
        ],
    },
    "Attendance": {
        "neg": [
            "meri attendance galat lagi hui hai",
            "present hone ke bawajood portal pe absent show ho raha hai",
            "attendance 75 percent se kam show ho rahi hai halanke main classes attend karta hoon",
            "attendance shortage ki wajah se exam mein nahi baithne de rahe",
            "teacher ne meri attendance mark nahi ki",
            "medical leave ki attendance abhi tak adjust nahi hui",
            "attendance sheet mein mera naam nahi hai",
            "attendance update nahi ho rahi",
            "my attendance is marked wrong in the portal",
            "attendance shortage condone nahi ki ja rahi",
        ],
        "neu": [
            "attendance ka minimum percentage kitna hona chahiye",
            "medical leave ki application kaise dein",
            "attendance shortage condone kaise hoti hai",
            "attendance kahan check kar sakte hain",
            "attendance ka record kab update hota hai",
            "where do I submit a leave application",
        ],
        "pos": [
            "attendance system ab bohat transparent hai",
            "attendance issue solve karne ka shukriya",
            "teacher ne attendance sahi kar di shukriya",
        ],
    },
    "Technical": {
        "neg": [
            "university ka wifi bilkul kaam nahi kar raha",
            "student portal open nahi ho raha",
            "email account login nahi ho raha",
            "lab ke computers kharab hain",
            "lab mein projector kaam nahi kar raha",
            "university website bohat slow chal rahi hai",
            "portal login ke liye OTP nahi aa raha",
            "portal pe error 500 aa raha hai",
            "the campus wifi is not working at all",
            "software lab mein system baar baar hang ho jata hai",
        ],
        "neu": [
            "wifi ka password kaise milega",
            "university email kaise banate hain",
            "lab mein software install kaise hoga",
            "portal ka username kya hota hai",
            "VPN access kaise milega",
            "where is the printing facility available",
        ],
        "pos": [
            "wifi ab bohat fast hai shukriya",
            "lab ke naye computers bohat acche hain",
            "IT department ne jaldi issue solve kar diya shukriya",
        ],
    },
    "General Query": {
        "neg": [
            "canteen mein khana bohat ganda hai",
            "library mein ac kaam nahi kar raha",
            "bus ka time table follow nahi hota",
            "hostel mein paani nahi aa raha",
            "class room mein bohat shor hota hai",
            "parking mein jagah nahi milti",
            "washroom mein safai nahi hoti",
            "teacher bohat late aate hain",
            "the library is always overcrowded",
        ],
        "neu": [
            "library ki timing kya hai",
            "transcript ke liye kaise apply karte hain",
            "student ID card kahan se milega",
            "administration office kab khulta hai",
            "degree verification kaise hoti hai",
            "how do I get a bus pass",
        ],
        "pos": [
            "library ka mahol bohat acha hai",
            "teachers ka behavior bohat acha hai shukriya",
            "campus ka environment bohat acha hai",
        ],
    },
}

OPENERS = ["", "", "sir ", "mam ", "assalam o alaikum sir, ", "hello sir, ",
           "respected sir, ", "dear sir/mam, ", "salam sir ", "sir please ", "Sir, "]

URGENT_SUFFIX = [
    " aur deadline kal hai", " aur aaj last date hai", " please jaldi solve kijiye",
    " urgent hai", " aur kal exam hai", " abhi ke abhi chahiye", " aaj hi resolve karwa dein",
    " aur deadline tonight hai", " bohat zaroori hai please jaldi dekhein",
    " and the deadline is tomorrow", " aur kal last date hai please help",
]
MEDIUM_SUFFIX = [
    "", "", "", " please check kar lein", " koi solution batayein",
    " kab tak theek hoga?", " please guide karein", " please help",
]
QUERY_SUFFIX = ["", "", "", "?", " please batayein", " koi guide kar dein"]
POS_SUFFIX = ["", "", " thank you", " shukriya", " keep it up"]
# Neutral query with mild time pressure -> Medium priority
QUERY_TIME_SUFFIX = [" kal tak batana hai", " is hafte mein chahiye", " jald batayein"]

# --------------------------------------------------------------------------
# Spelling noise (simulates inconsistent Roman Urdu spelling)
# --------------------------------------------------------------------------
NOISE = {
    "nahi": ["nahi", "ni", "nai", "nhi", "nahin"],
    "hai": ["hai", "hai", "hai", "h", "hy"],
    "aur": ["aur", "aur", "or", "aor"],
    "mein": ["mein", "me", "main"],
    "ho raha": ["ho raha", "horaha", "hora", "ho rha"],
    "ho rahi": ["ho rahi", "horahi", "hori", "ho rhi"],
    "please": ["please", "plz", "pls"],
    "bohat": ["bohat", "bhot", "bohot", "boht"],
    "kaise": ["kaise", "kese", "kaisay"],
    "shukriya": ["shukriya", "shukria", "thanks"],
    "pe": ["pe", "py", "par", "pr"],
    "kal": ["kal", "kl"],
    "aaj": ["aaj", "aj"],
    "kya": ["kya", "kia"],
    "attendance": ["attendance", "attendence"],
    "assignment": ["assignment", "asignment"],
    "registration": ["registration", "regestration"],
}


def add_noise(text: str, p: float = 0.5) -> str:
    out = text
    for canon, variants in NOISE.items():
        if canon in out and random.random() < p:
            out = out.replace(canon, random.choice(variants))
    r = random.random()
    if r < 0.12:
        out = out.upper()
    elif r < 0.20:
        out = out.capitalize()
    if random.random() < 0.15:
        out += random.choice(["!!", "!!!", "..", " :("])
    return out


def build_sentence(cat, sent):
    base = random.choice(ISSUES[cat][sent])
    opener = random.choice(OPENERS)
    if sent == "neg":
        if random.random() < 0.40:
            suffix, prio = random.choice(URGENT_SUFFIX), "High"
        else:
            suffix, prio = random.choice(MEDIUM_SUFFIX), "Medium"
        label = "Negative"
    elif sent == "neu":
        if random.random() < 0.15:
            suffix, prio = random.choice(QUERY_TIME_SUFFIX), "Medium"
        else:
            suffix, prio = random.choice(QUERY_SUFFIX), "Low"
        label = "Neutral"
    else:
        suffix, prio = random.choice(POS_SUFFIX), "Low"
        label = "Positive"
    text = add_noise(f"{opener}{base}{suffix}")
    return text, label, prio


def build_train(n_per_cat=130):
    rows, seen = [], set()
    plan = {"neg": 0.55, "neu": 0.32, "pos": 0.13}
    for cat in ISSUES:
        for sent, share in plan.items():
            target = int(n_per_cat * share)
            tries = 0
            got = 0
            while got < target and tries < target * 40:
                tries += 1
                text, sentiment, prio = build_sentence(cat, sent)
                key = text.lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                rows.append((text, cat, sentiment, prio))
                got += 1
    df = pd.DataFrame(rows, columns=["text", "category", "sentiment", "priority"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# --------------------------------------------------------------------------
# Hand-written natural test set: wording intentionally different from templates
# (text, category, sentiment, priority)
# --------------------------------------------------------------------------
NATURAL = [
    # LMS
    ("sir LMS pe assignment submit ni horahi aur deadline kal hai", "LMS", "Negative", "High"),
    ("lms ma mera course dikh hi nhi raha pichle 2 din se", "LMS", "Negative", "Medium"),
    ("mam quiz attempt karte hue lms crash ho gaya aur marks zero aa gaye please fix urgent", "LMS", "Negative", "High"),
    ("lms par uploaded pdf open nahi hoti koi dekh lein", "LMS", "Negative", "Medium"),
    ("lecture 5 ki video lms pe kab lagegi", "LMS", "Neutral", "Low"),
    ("lms ka naya dashboard kafi achha lag raha hai", "LMS", "Positive", "Low"),
    ("lms mein group assignment ka option kahan hota hai", "LMS", "Neutral", "Low"),
    ("sir lms ka link kharab hai assignment aaj raat 12 baje tak jama karni hai", "LMS", "Negative", "High"),
    ("lms pe discussion forum open nhi ho raha", "LMS", "Negative", "Medium"),
    ("lms ke zariye attendance bhi dekhi ja sakti hai kya", "LMS", "Neutral", "Low"),
    # Fee
    ("fee jama krwa di thi 3 din pehle lekin portal abhi bhi unpaid dikha raha hai", "Fee", "Negative", "Medium"),
    ("kal fee ki last date hai aur challan generate nhi ho raha please help", "Fee", "Negative", "High"),
    ("fee waiver ka form kahan se milega", "Fee", "Neutral", "Low"),
    ("mujhse late fee charge ki gayi jabke maine time par pay kiya tha", "Fee", "Negative", "Medium"),
    ("installment wali facility bohat madadgar thi thanks accounts office", "Fee", "Positive", "Low"),
    ("bank ne amount kaat liya magar university ko receive nahi hua aaj hi check karwa dein", "Fee", "Negative", "High"),
    ("next semester ki fee kitni hogi", "Fee", "Neutral", "Low"),
    ("scholarship ki rakam abhi tak adjust nahi hui", "Fee", "Negative", "Medium"),
    ("fee voucher pe mera naam galat print hua hai", "Fee", "Negative", "Medium"),
    ("shukriya sir fee refund mil gaya", "Fee", "Positive", "Low"),
    # Registration
    ("registration open hote hi portal hang ho jata hai aur kal last date hai", "Registration", "Negative", "High"),
    ("kya main apna elective course badal sakta hoon", "Registration", "Neutral", "Low"),
    ("mera section full ho gaya hai aur koi alternative nahi mil raha", "Registration", "Negative", "Medium"),
    ("course registration ka process bohat asaan tha thanks", "Registration", "Positive", "Low"),
    ("add drop ki window kab band hogi", "Registration", "Neutral", "Low"),
    ("registration submit krne ke baad bhi confirmation email nhi aaya", "Registration", "Negative", "Medium"),
    ("sir mera prerequisite pass hai phir bhi system course allow nahi kar raha aaj last date hai", "Registration", "Negative", "High"),
    ("semester freeze ki application ka status kahan check karun", "Registration", "Neutral", "Low"),
    ("registration office ne bohat jaldi kaam kar diya shukria", "Registration", "Positive", "Low"),
    ("ek hi course do dafa registered show ho raha hai", "Registration", "Negative", "Medium"),
    # Examination
    ("sir kal mera paper hai aur admit card abhi tak nahi mila", "Examination", "Negative", "High"),
    ("result mein mera ek course missing hai", "Examination", "Negative", "Medium"),
    ("final exam ki date sheet kab tak aa jayegi", "Examination", "Neutral", "Low"),
    ("do papers ek hi time pe rakhe gaye hain please reschedule krwa dein", "Examination", "Negative", "Medium"),
    ("paper ki checking bohat achi hui thanks sir", "Examination", "Positive", "Low"),
    ("rechecking fee kitni hoti hai", "Examination", "Neutral", "Low"),
    ("aaj ka paper out of course tha students ke saath na-insafi hui hai", "Examination", "Negative", "Medium"),
    ("marks ki entry mein ghalti hai gpa kam ho gaya aaj hi correct karwa dein", "Examination", "Negative", "High"),
    ("exam hall mein bohat garmi thi", "Examination", "Negative", "Low"),
    ("makeup exam ke liye kis ko application deni hai", "Examination", "Neutral", "Low"),
    # Attendance
    ("mujhe class mein present hone ke bawajood absent mark kiya gaya hai", "Attendance", "Negative", "Medium"),
    ("sir attendance shortage ki wajah se kal ka paper nahi dene de rahe please urgent help", "Attendance", "Negative", "High"),
    ("attendance sheet kahan dekh sakte hain", "Attendance", "Neutral", "Low"),
    ("medical certificate jama kiya tha lekin attendance abhi bhi kam dikha rahi hai", "Attendance", "Negative", "Medium"),
    ("attendance ka naya system acha hai shukriya", "Attendance", "Positive", "Low"),
    ("condonation ki application kab tak de sakte hain", "Attendance", "Neutral", "Low"),
    ("sir ne meri 5 classes ki attendance lagayi hi nahi", "Attendance", "Negative", "Medium"),
    ("attendance 70 percent hai kya mujhe exam allow hoga", "Attendance", "Neutral", "Medium"),
    ("attendance portal pe kal se update nhi ho raha", "Attendance", "Negative", "Medium"),
    ("aaj attendance short hone ki wajah se roll number slip rok li gayi hai please jaldi dekhein", "Attendance", "Negative", "High"),
    # Technical
    ("campus ka wifi subah se down hai kuch load nahi hota", "Technical", "Negative", "Medium"),
    ("portal login ka otp nahi aa raha aur aaj form submit karna hai", "Technical", "Negative", "High"),
    ("lab ke pc itne slow hain ke kaam hi nahi hota", "Technical", "Negative", "Medium"),
    ("university email ka password bhool gaya hoon kaise reset karun", "Technical", "Neutral", "Low"),
    ("naya wifi router lagane ka shukriya speed bohat achi hai", "Technical", "Positive", "Low"),
    ("student portal pe 404 error aa raha hai", "Technical", "Negative", "Medium"),
    ("lab mein python install karwane ke liye kis se rabta karun", "Technical", "Neutral", "Low"),
    ("projector band hai class kaise hogi abhi", "Technical", "Negative", "High"),
    ("software license expire ho gaya hai lab ke computers mein", "Technical", "Negative", "Medium"),
    ("wifi ka password kya hai", "Technical", "Neutral", "Low"),
    # General Query
    ("library ki book issue karne ki limit kitni hai", "General Query", "Neutral", "Low"),
    ("canteen ki quality din ba din kharab ho rahi hai", "General Query", "Negative", "Medium"),
    ("campus ka mahol bohat pur sukoon hai", "General Query", "Positive", "Low"),
    ("degree kab tak mil jati hai convocation ke baad", "General Query", "Neutral", "Low"),
    ("bus subah late aati hai jis se classes miss hoti hain", "General Query", "Negative", "Medium"),
    ("hostel mein 2 din se bijli nahi hai", "General Query", "Negative", "High"),
    ("student card kho gaya hai kya karna hoga", "General Query", "Neutral", "Low"),
    ("sports complex ki timings kya hain", "General Query", "Neutral", "Low"),
    ("library staff ka rawayya bohat acha hai shukriya", "General Query", "Positive", "Low"),
    ("class rooms ka ac kharab hai aur garmi mein bohat masla ho raha hai", "General Query", "Negative", "Medium"),
]


def main():
    train = build_train()
    test = pd.DataFrame(NATURAL, columns=["text", "category", "sentiment", "priority"])
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test_natural.csv", index=False)
    print(f"train: {len(train)} rows, natural test: {len(test)} rows")
    for col in ["category", "sentiment", "priority"]:
        print(f"\n[train] {col}\n{train[col].value_counts().to_string()}")
    print("\n[natural test] category\n" + test["category"].value_counts().to_string())
    print("\n[natural test] sentiment\n" + test["sentiment"].value_counts().to_string())
    print("\n[natural test] priority\n" + test["priority"].value_counts().to_string())


if __name__ == "__main__":
    main()
