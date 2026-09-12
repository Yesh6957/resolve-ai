import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EVALUATION_RESULTS_PATH = (
    PROCESSED_DIR / "evaluation_results.csv"
)

BASELINE_RESULTS_PATH = (
    PROCESSED_DIR / "baseline_results.txt"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("📈 TRUE BASELINES VS AGENT PERFORMANCE")
print("=" * 60)


# ============================================================
# CHECK EVALUATION FILE
# ============================================================

print("\n📁 Looking for evaluation results:")
print(EVALUATION_RESULTS_PATH)

if not EVALUATION_RESULTS_PATH.exists():

    print("\n❌ evaluation_results.csv was not found.")

    print("\nRun this first:")
    print("python src/evaluation.py")

    raise FileNotFoundError(
        f"\nCould not find:\n{EVALUATION_RESULTS_PATH}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(EVALUATION_RESULTS_PATH)

print(
    f"\n✅ Loaded {len(df)} evaluation records."
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "customer_text",
    "true_intent",
    "pred_intent",
    "true_escalate",
    "pred_escalate"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\n❌ Missing required columns:")

    for column in missing_columns:
        print(f"   - {column}")

    print("\nAvailable columns:")

    for column in df.columns:
        print(f"   - {column}")

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# BOOLEAN NORMALIZATION
# ============================================================

def normalize_bool(value):
    """
    Convert common CSV boolean representations
    into real Python booleans.
    """

    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in {
        "true",
        "1",
        "yes",
        "y",
        "t"
    }:
        return True

    if value in {
        "false",
        "0",
        "no",
        "n",
        "f"
    }:
        return False

    return False


# ============================================================
# CLEAN INTENT DATA
# ============================================================

intent_df = df[
    df["true_intent"].notna()
    &
    df["pred_intent"].notna()
].copy()

intent_df["true_intent"] = (
    intent_df["true_intent"]
    .astype(str)
    .str.strip()
    .str.lower()
)

intent_df["pred_intent"] = (
    intent_df["pred_intent"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# CLEAN ESCALATION DATA
# ============================================================

escalation_df = df[
    df["true_escalate"].notna()
    &
    df["pred_escalate"].notna()
].copy()

escalation_df["true_escalate"] = (
    escalation_df["true_escalate"]
    .apply(normalize_bool)
)

escalation_df["pred_escalate"] = (
    escalation_df["pred_escalate"]
    .apply(normalize_bool)
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("📊 DATASET INFORMATION")
print("=" * 60)

print(f"\nTotal evaluation rows: {len(df)}")

print(
    f"Valid intent predictions: "
    f"{len(intent_df)}"
)

print(
    f"Valid escalation predictions: "
    f"{len(escalation_df)}"
)


# ============================================================
# TRUE INTENT DISTRIBUTION
# ============================================================

print("\n📊 True Intent Distribution:")

intent_distribution = (
    intent_df["true_intent"]
    .value_counts()
)

for intent, count in intent_distribution.items():

    percentage = count / len(intent_df)

    print(
        f"  {intent}: "
        f"{count} "
        f"({percentage:.2%})"
    )


# ============================================================
# 1. TRIVIAL INTENT BASELINE
# ============================================================

print("\n" + "=" * 60)
print("1️⃣ TRIVIAL BASELINE")
print("=" * 60)

majority_intent = (
    intent_df["true_intent"]
    .mode()[0]
)

trivial_intent_preds = [
    majority_intent
] * len(intent_df)

trivial_accuracy = accuracy_score(
    intent_df["true_intent"],
    trivial_intent_preds
)

print(
    f"\nMost frequent intent: "
    f"{majority_intent}"
)

print(
    f"🎯 Trivial Baseline Intent Accuracy: "
    f"{trivial_accuracy:.2%}"
)


# ============================================================
# 2. SIMPLE KEYWORD BASELINE
# ============================================================

print("\n" + "=" * 60)
print("2️⃣ SIMPLE KEYWORD BASELINE")
print("=" * 60)


def keyword_intent(text):
    """
    Simple non-LLM keyword baseline.

    Intentionally deterministic and explainable.
    """

    text = (
        str(text)
        .lower()
        .strip()
    )

    # --------------------------------------------------------
    # ACCOUNT SECURITY
    # --------------------------------------------------------

    account_keywords = [
        "password",
        "forgot password",
        "reset password",
        "locked out",
        "account locked",
        "login",
        "log in",
        "sign in",
        "apple id",
        "apple account",
        "account recovery",
        "can't access my account",
        "cannot access my account"
    ]

    if any(
        keyword in text
        for keyword in account_keywords
    ):
        return "account_security"


    # --------------------------------------------------------
    # HARDWARE REPAIR
    # --------------------------------------------------------

    hardware_keywords = [
        "cracked screen",
        "broken screen",
        "screen cracked",
        "screen broken",
        "cracked display",
        "broken display",
        "water damage",
        "water damaged",
        "dropped my iphone",
        "dropped my phone",
        "physical damage",
        "damaged iphone",
        "damaged phone",
        "swollen battery",
        "battery swollen",
        "battery bulging",
        "won't turn on",
        "wont turn on",
        "doesn't turn on",
        "doesnt turn on"
    ]

    if any(
        keyword in text
        for keyword in hardware_keywords
    ):
        return "hardware_repair"


    # --------------------------------------------------------
    # BILLING / PURCHASES
    # --------------------------------------------------------

    billing_keywords = [
        "refund",
        "refund me",
        "charged",
        "charge",
        "charged twice",
        "wrong charge",
        "incorrect charge",
        "unknown charge",
        "unauthorized charge",
        "billing",
        "bill",
        "payment",
        "purchase",
        "purchased",
        "subscription",
        "money back",
        "payment issue",
        "app i didn't buy",
        "app i did not buy"
    ]

    if any(
        keyword in text
        for keyword in billing_keywords
    ):
        return "billing_purchases"


    # --------------------------------------------------------
    # DEVICE CONNECTIVITY
    # --------------------------------------------------------

    connectivity_keywords = [
        "wifi",
        "wi-fi",
        "wireless",
        "bluetooth",
        "hotspot",
        "cellular",
        "mobile data",
        "network",
        "connection",
        "connect",
        "connecting",
        "airdrop",
        "internet connection",
        "can't connect",
        "cannot connect",
        "won't connect",
        "wont connect"
    ]

    if any(
        keyword in text
        for keyword in connectivity_keywords
    ):
        return "device_connectivity"


    # --------------------------------------------------------
    # SOFTWARE TROUBLESHOOTING
    # --------------------------------------------------------

    software_keywords = [
        "software",
        "ios",
        "ios update",
        "update",
        "latest update",
        "bug",
        "glitch",
        "crash",
        "crashing",
        "crashed",
        "freezing",
        "frozen",
        "app not working",
        "application not working",
        "keeps crashing",
        "keeps freezing",
        "system error",
        "error message"
    ]

    if any(
        keyword in text
        for keyword in software_keywords
    ):
        return "software_troubleshooting"


    # --------------------------------------------------------
    # OTHER
    # --------------------------------------------------------

    # We intentionally don't force uncertain examples
    # into a more specific technical category.

    other_keywords = [
        "legal",
        "lawsuit",
        "press",
        "media",
        "complaint",
        "feedback",
        "suggestion"
    ]

    if any(
        keyword in text
        for keyword in other_keywords
    ):
        return "other"


    # --------------------------------------------------------
    # GENERAL INQUIRY
    # --------------------------------------------------------

    return "general_inquiry"


df["heuristic_intent"] = (
    df["customer_text"]
    .fillna("")
    .apply(keyword_intent)
)


heuristic_valid = df[
    df["true_intent"].notna()
].copy()

heuristic_valid["true_intent"] = (
    heuristic_valid["true_intent"]
    .astype(str)
    .str.strip()
    .str.lower()
)

heuristic_accuracy = accuracy_score(
    heuristic_valid["true_intent"],
    heuristic_valid["heuristic_intent"]
)

print(
    f"\n🎯 Simple Heuristic Intent Accuracy: "
    f"{heuristic_accuracy:.2%}"
)


# ============================================================
# 3. AGENT INTENT PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("3️⃣ AGENT PERFORMANCE")
print("=" * 60)

agent_accuracy = accuracy_score(
    intent_df["true_intent"],
    intent_df["pred_intent"]
)

print(
    f"\n🤖 Agent Intent Accuracy: "
    f"{agent_accuracy:.2%}"
)


# ============================================================
# AGENT PER-CLASS PERFORMANCE
# ============================================================

print("\n📋 Agent Per-Intent Performance:")

agent_report = classification_report(
    intent_df["true_intent"],
    intent_df["pred_intent"],
    zero_division=0
)

print(agent_report)


# ============================================================
# HEURISTIC PER-CLASS PERFORMANCE
# ============================================================

print("\n📋 Heuristic Per-Intent Performance:")

heuristic_report = classification_report(
    heuristic_valid["true_intent"],
    heuristic_valid["heuristic_intent"],
    zero_division=0
)

print(heuristic_report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("🔍 AGENT CONFUSION MATRIX")
print("=" * 60)

labels = sorted(
    set(intent_df["true_intent"])
    |
    set(intent_df["pred_intent"])
)

cm = confusion_matrix(
    intent_df["true_intent"],
    intent_df["pred_intent"],
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"TRUE: {label}"
        for label in labels
    ],
    columns=[
        f"PRED: {label}"
        for label in labels
    ]
)

print(cm_df)


# ============================================================
# 4. ESCALATION BASELINES
# ============================================================

print("\n" + "=" * 60)
print("🚨 ESCALATION BASELINES VS AGENT")
print("=" * 60)


# ------------------------------------------------------------
# TRIVIAL ESCALATION BASELINE
# ------------------------------------------------------------

majority_escalation = (
    escalation_df["true_escalate"]
    .mode()[0]
)

trivial_escalation_preds = [
    majority_escalation
] * len(escalation_df)

trivial_escalation_accuracy = accuracy_score(
    escalation_df["true_escalate"],
    trivial_escalation_preds
)

print(
    f"\nTrivial Escalation Baseline Accuracy: "
    f"{trivial_escalation_accuracy:.2%}"
)

print(
    f"Most frequent escalation class: "
    f"{majority_escalation}"
)


# ------------------------------------------------------------
# SIMPLE ESCALATION HEURISTIC
# ------------------------------------------------------------

def heuristic_escalation(text):

    text = (
        str(text)
        .lower()
        .strip()
    )

    escalation_keywords = [

        # Billing / refunds
        "refund",
        "money back",
        "charged twice",
        "wrong charge",
        "incorrect charge",
        "unauthorized charge",
        "fraud",
        "dispute",
        "charge i don't recognize",
        "charge i do not recognize",

        # Hardware damage
        "cracked screen",
        "broken screen",
        "screen cracked",
        "screen broken",
        "water damage",
        "water damaged",
        "swollen battery",
        "battery swollen",
        "physical damage",
        "dropped my iphone",
        "dropped my phone",

        # Strong frustration
        "fuck",
        "fucking",
        "shit",
        "terrible",
        "horrible",
        "furious",
        "angry",
        "ridiculous",
        "worst",
        "unacceptable",

        # PII
        "@gmail.com",
        "@yahoo.com",
        "@outlook.com",
        "@hotmail.com",
        "phone number",
        "my number is",
        "my email is"
    ]

    return any(
        keyword in text
        for keyword in escalation_keywords
    )


df["heuristic_escalate"] = (
    df["customer_text"]
    .fillna("")
    .apply(heuristic_escalation)
)


heuristic_escalation_valid = df[
    df["true_escalate"].notna()
].copy()

heuristic_escalation_valid["true_escalate"] = (
    heuristic_escalation_valid["true_escalate"]
    .apply(normalize_bool)
)


heuristic_escalation_accuracy = accuracy_score(
    heuristic_escalation_valid["true_escalate"],
    heuristic_escalation_valid["heuristic_escalate"]
)

print(
    f"\nSimple Heuristic Escalation Accuracy: "
    f"{heuristic_escalation_accuracy:.2%}"
)


# ============================================================
# AGENT ESCALATION PERFORMANCE
# ============================================================

agent_escalation_accuracy = accuracy_score(
    escalation_df["true_escalate"],
    escalation_df["pred_escalate"]
)

agent_escalation_precision = precision_score(
    escalation_df["true_escalate"],
    escalation_df["pred_escalate"],
    zero_division=0
)

agent_escalation_recall = recall_score(
    escalation_df["true_escalate"],
    escalation_df["pred_escalate"],
    zero_division=0
)

agent_escalation_f1 = f1_score(
    escalation_df["true_escalate"],
    escalation_df["pred_escalate"],
    zero_division=0
)

print(
    f"\n🤖 Agent Escalation Accuracy: "
    f"{agent_escalation_accuracy:.2%}"
)

print(
    f"🤖 Agent Escalation Precision: "
    f"{agent_escalation_precision:.2%}"
)

print(
    f"🤖 Agent Escalation Recall: "
    f"{agent_escalation_recall:.2%}"
)

print(
    f"🤖 Agent Escalation F1: "
    f"{agent_escalation_f1:.2%}"
)


# ============================================================
# REPLY GENERATION
# ============================================================

print("\n" + "=" * 60)
print("💬 REPLY GENERATION")
print("=" * 60)

if "drafted_reply" in df.columns:

    replies = df[
        df["drafted_reply"].notna()
        &
        (
            df["drafted_reply"]
            .astype(str)
            .str.strip()
            .str.len()
            > 0
        )
    ]

    reply_rate = (
        len(replies) / len(df)
        if len(df) > 0
        else 0
    )

    print(
        f"\nGenerated replies: "
        f"{len(replies)}/{len(df)}"
    )

    print(
        f"Reply generation rate: "
        f"{reply_rate:.2%}"
    )

else:

    print(
        "\n⚠️ drafted_reply column not found."
    )


# ============================================================
# LLM JUDGE
# ============================================================

print("\n" + "=" * 60)
print("🤖 LLM JUDGE")
print("=" * 60)


if "judge_tone" in df.columns:

    tone_scores = pd.to_numeric(
        df["judge_tone"],
        errors="coerce"
    )

    valid_tone = tone_scores.dropna()

    if not valid_tone.empty:

        print(
            f"\nAverage Tone Score: "
            f"{valid_tone.mean():.2f}/5"
        )

        print(
            f"Judge Tone Samples: "
            f"{len(valid_tone)}"
        )

    else:

        print(
            "\nAverage Tone Score: "
            "No scores available"
        )

else:

    print(
        "\n⚠️ judge_tone column not found."
    )


if "judge_grounding" in df.columns:

    grounding_scores = pd.to_numeric(
        df["judge_grounding"],
        errors="coerce"
    )

    valid_grounding = grounding_scores.dropna()

    if not valid_grounding.empty:

        print(
            f"Average Grounding Score: "
            f"{valid_grounding.mean():.2f}/5"
        )

        print(
            f"Judge Grounding Samples: "
            f"{len(valid_grounding)}"
        )

    else:

        print(
            "\nAverage Grounding Score: "
            "No scores available"
        )

else:

    print(
        "\n⚠️ judge_grounding column not found."
    )


# ============================================================
# FINAL INTENT COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("🏆 FINAL INTENT COMPARISON")
print("=" * 60)

print(
    f"\nTrivial Baseline:      "
    f"{trivial_accuracy:.2%}"
)

print(
    f"Simple Heuristic:      "
    f"{heuristic_accuracy:.2%}"
)

print(
    f"Agent (Codestral LLM): "
    f"{agent_accuracy:.2%}"
)


# ============================================================
# FINAL ESCALATION COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("🏆 FINAL ESCALATION COMPARISON")
print("=" * 60)

print(
    f"\nTrivial Baseline:      "
    f"{trivial_escalation_accuracy:.2%}"
)

print(
    f"Simple Heuristic:      "
    f"{heuristic_escalation_accuracy:.2%}"
)

print(
    f"Agent Accuracy:        "
    f"{agent_escalation_accuracy:.2%}"
)

print(
    f"Agent Precision:       "
    f"{agent_escalation_precision:.2%}"
)

print(
    f"Agent Recall:          "
    f"{agent_escalation_recall:.2%}"
)

print(
    f"Agent F1:              "
    f"{agent_escalation_f1:.2%}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_lines = [

    "TRUE BASELINES VS AGENT PERFORMANCE",

    "",

    "DATASET",

    f"Evaluation rows: {len(df)}",

    f"Valid intent predictions: {len(intent_df)}",

    f"Valid escalation predictions: "
    f"{len(escalation_df)}",

    "",

    "INTENT",

    f"Majority intent: {majority_intent}",

    f"Trivial Baseline Accuracy: "
    f"{trivial_accuracy:.4f}",

    f"Simple Heuristic Accuracy: "
    f"{heuristic_accuracy:.4f}",

    f"Agent Accuracy: "
    f"{agent_accuracy:.4f}",

    "",

    "ESCALATION",

    f"Majority escalation class: "
    f"{majority_escalation}",

    f"Trivial Baseline Accuracy: "
    f"{trivial_escalation_accuracy:.4f}",

    f"Simple Heuristic Accuracy: "
    f"{heuristic_escalation_accuracy:.4f}",

    f"Agent Accuracy: "
    f"{agent_escalation_accuracy:.4f}",

    f"Agent Precision: "
    f"{agent_escalation_precision:.4f}",

    f"Agent Recall: "
    f"{agent_escalation_recall:.4f}",

    f"Agent F1: "
    f"{agent_escalation_f1:.4f}",
]


if "drafted_reply" in df.columns:

    summary_lines.extend([
        "",
        "REPLIES",
        f"Generated replies: {len(replies)}",
        f"Reply generation rate: {reply_rate:.4f}"
    ])


if "judge_tone" in df.columns:

    tone_scores = pd.to_numeric(
        df["judge_tone"],
        errors="coerce"
    ).dropna()

    if not tone_scores.empty:

        summary_lines.append(
            f"Average Tone Score: "
            f"{tone_scores.mean():.4f}"
        )


if "judge_grounding" in df.columns:

    grounding_scores = pd.to_numeric(
        df["judge_grounding"],
        errors="coerce"
    ).dropna()

    if not grounding_scores.empty:

        summary_lines.append(
            f"Average Grounding Score: "
            f"{grounding_scores.mean():.4f}"
        )


# ============================================================
# WRITE SUMMARY
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    BASELINE_RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary_lines)
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("✅ Baseline evaluation completed.")
print("=" * 60)

print(
    f"\n📁 Summary saved to:\n"
    f"{BASELINE_RESULTS_PATH}"
)

print("=" * 60)
