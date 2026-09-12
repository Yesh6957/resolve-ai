from src.pipeline import SupportAgent

if __name__ == "__main__":
    # Initialize agent
    agent = SupportAgent()
    
    # 1. Build the database (only runs once)
    agent.build_knowledge_base()
    
    # 2. Test an Escalation Scenario (Hardware Damage)
    print("\n--- Test 1: Hardware Issue ---")
    test_tweet_1 = "I just dropped my iPhone and the screen is completely shattered. How do I fix this?"
    result_1 = agent.process_tweet(test_tweet_1)
    print(result_1)
    
    # 3. Test an Auto-Handle Scenario (Software Troubleshooting)
    print("\n--- Test 2: Software Issue ---")
    test_tweet_2 = "My Apple Music keeps freezing every time I try to play a downloaded playlist on iOS 17."
    result_2 = agent.process_tweet(test_tweet_2)
    print(result_2)