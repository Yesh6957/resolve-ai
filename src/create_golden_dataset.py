import pandas as pd
import os

PROCESSED_PATH = '../data/processed/clean_brand_threads.csv'
GOLDEN_PATH = '../data/processed/golden_dataset.csv'
RAG_DB_PATH = '../data/processed/rag_knowledge_base.csv'

intents = {
    '1': 'account_security', '2': 'hardware_repair', 
    '3': 'software_troubleshooting', '4': 'billing_purchases',
    '5': 'device_connectivity', '6': 'general_inquiry', '7': 'other'
}

def setup_data():
    df = pd.read_csv(PROCESSED_PATH)
    # Shuffle again just to be safe
    df = df.sample(frac=1, random_state=101).reset_index(drop=True)
    
    # STRICT SPLIT: 200 for testing, the rest for RAG
    df_golden = df.iloc[:200].copy()
    df_rag = df.iloc[200:].copy()
    
    df_rag.to_csv(RAG_DB_PATH, index=False)
    print(f"✅ Saved {len(df_rag)} rows to RAG Knowledge Base.")
    
    # Add empty columns for our labels
    df_golden['true_intent'] = None
    df_golden['should_escalate'] = None
    df_golden['escalation_reason'] = None
    
    return df_golden

def interactive_labeler(df):
    print("\n🚀 Starting Rapid Labeling CLI 🚀")
    print("Press CTRL+C at any time to save and exit.\n")
    
    try:
        for idx, row in df.iterrows():
            # Skip already labeled rows (so you can resume later)
            if pd.notna(row['true_intent']):
                continue
            
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Tweet {idx + 1} / 200 ---")
            print(f"CUSTOMER: {row['customer_text']}\n")
            
            # 1. Intent Labeling
            for k, v in intents.items():
                print(f"[{k}] {v}")
            
            intent_choice = ''
            while intent_choice not in intents:
                intent_choice = input("\nSelect Intent (1-7): ")
            df.at[idx, 'true_intent'] = intents[intent_choice]
            
            # 2. Escalation Labeling
            escalate_choice = input("Escalate to human? (y/n): ").lower()
            if escalate_choice == 'y':
                df.at[idx, 'should_escalate'] = 1
                reason = input("Reason (e.g., 'billing', 'hardware_damage', 'toxicity'): ")
                df.at[idx, 'escalation_reason'] = reason
            else:
                df.at[idx, 'should_escalate'] = 0
                df.at[idx, 'escalation_reason'] = 'Safe for auto-reply'
            
            # Auto-save after every row so you don't lose progress
            df.to_csv(GOLDEN_PATH, index=False)
            
    except KeyboardInterrupt:
        print("\n💾 Saved progress safely. See you next time!")
    
    print("\n🎉 Labeling Complete! golden_dataset.csv is ready.")

if __name__ == "__main__":
    # Load existing progress if available, otherwise setup fresh
    if os.path.exists(GOLDEN_PATH):
        df_golden = pd.read_csv(GOLDEN_PATH)
    else:
        df_golden = setup_data()
        
    interactive_labeler(df_golden)