import pandas as pd
from datasets import load_dataset
import os

def load_and_filter_data():
    print("Downloading CLINC150 dataset from Hugging Face...")
    dataset = load_dataset("clinc_oos", "plus")
    
    selected_intents = [
        "order_status", "cancel_order", "change_order", "damaged_receipt",
        "report_fraud", "report_lost_card", "replacement_card_duration",
        "pin_change", "expiration_date", "bill_balance", "bill_due",
        "pay_bill", "routing", "direct_deposit", "balance", "transfer",
        "rewards_balance", "redeem_rewards", "credit_limit", 
        "credit_limit_change", "apr", "application_status", "account_blocked",
        "update_password", "freeze_account"
    ]
    
    # Map the string intents back to their integer IDs in the dataset
    intent_names = dataset['train'].features['intent'].names
    selected_intent_ids = {intent_names.index(intent): intent for intent in selected_intents if intent in intent_names}
    
    oos_id = intent_names.index("oos")
    selected_intent_ids[oos_id] = "oos"

    def filter_and_format(split_name):
        df = dataset[split_name].to_pandas()
        filtered_df = df[df['intent'].isin(selected_intent_ids.keys())].copy()
        # Map integer labels back to readable strings
        filtered_df['intent_name'] = filtered_df['intent'].map(selected_intent_ids)
        return filtered_df

    print("Filtering down to 25 customer service intents + Out-of-Scope data...")
    train_df = filter_and_format("train")
    val_df = filter_and_format("validation")
    test_df = filter_and_format("test")
    
    # Save the processed datasets to ensure both models use the exact same data
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"Data saved successfully to data/processed/")
    print(f"Train size: {len(train_df)} | Validation size: {len(val_df)} | Test size: {len(test_df)}")

if __name__ == "__main__":
    load_and_filter_data() 
