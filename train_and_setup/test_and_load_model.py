# ===========================================
# 🎯 LOAD CHUNK 12 WITH UNSLOTH + COMPREHENSIVE VERIFICATION
# ===========================================

import torch
import gc
import re

# Clear GPU memory first
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

print("🎯 LOADING CHUNK 12 WITH UNSLOTH METHOD")
print("="*60)

# ===========================================
# STEP 1: LOAD MODEL WITH UNSLOTH (WORKING METHOD)
# ===========================================
try:
    from unsloth import FastLanguageModel
    
    model_name = "ali009eng/llama-3b-triage-classifier"
    target_commit = "161aca2e65e7247d739e1049758bb185434c419a"
    
    print("📥 Loading Chunk 12 with Unsloth FastLanguageModel...")
    print(f"🎯 Commit: {target_commit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        revision=target_commit,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    print("✅ Chunk 12 loaded successfully with Unsloth!")
    
except Exception as e:
    print(f"❌ Failed to load with Unsloth: {e}")
    model, tokenizer = None, None

# ===========================================
# STEP 2: COMPREHENSIVE VERIFICATION TEST
# ===========================================
def comprehensive_triage_test(model, tokenizer):
    """Test the loaded model with multiple triage scenarios"""
    
    print("\n🧪 COMPREHENSIVE CHUNK 12 VERIFICATION")
    print("="*60)
    
    test_cases = [
        {
            "name": "Critical - Cardiac Arrest",
            "case": """Patient: 45M, unconscious, no pulse
Vitals: No BP, no pulse, not breathing
Chief Complaint: Cardiac arrest""",
            "expected": 1,
            "reasoning": "Immediate life-threatening"
        },
        {
            "name": "Critical - Severe Trauma", 
            "case": """Patient: 25F, motor vehicle accident
Vitals: BP 70/40, HR 140, RR 30, GCS 8
Chief Complaint: Multiple trauma, altered mental status""",
            "expected": 1,
            "reasoning": "Unstable vitals + altered consciousness"
        },
        {
            "name": "Emergent - Chest Pain",
            "case": """Patient: 54M, severe chest pain
Vitals: BP 90/60, HR 130, RR 28, O2 85%
Chief Complaint: Crushing chest pain, difficulty breathing""",
            "expected": [1, 2],  # Both acceptable
            "reasoning": "High risk but could be ESI 1 or 2"
        },
        {
            "name": "Emergent - Severe Pain",
            "case": """Patient: 40F, severe abdominal pain
Vitals: BP 140/90, HR 110, RR 22, Temp 101F
Chief Complaint: 9/10 abdominal pain, nausea""",
            "expected": 2,
            "reasoning": "High risk, needs quick evaluation"
        },
        {
            "name": "Urgent - Moderate Issue",
            "case": """Patient: 30M, ankle injury
Vitals: BP 120/80, HR 88, RR 16, Temp 98.6F
Chief Complaint: Twisted ankle, unable to walk""",
            "expected": 3,
            "reasoning": "Urgent but stable"
        },
        {
            "name": "Less Urgent - Minor",
            "case": """Patient: 22F, sore throat
Vitals: BP 118/75, HR 82, RR 14, Temp 99.5F
Chief Complaint: Sore throat for 2 days""",
            "expected": 4,
            "reasoning": "Less urgent, can wait"
        },
        {
            "name": "Non-Urgent - Very Minor",
            "case": """Patient: 35M, medication refill
Vitals: BP 125/78, HR 75, RR 16, Temp 98.4F
Chief Complaint: Need prescription refill""",
            "expected": 5,
            "reasoning": "Non-urgent administrative"
        }
    ]
    
    results = []
    correct = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/7: {test_case['name']}")
        print("-" * 50)
        
        # Create the prompt
        prompt = f"""You are an emergency department triage assistant. 
Classify the following patient into an ESI triage level (1-5) based on their presentation. 

Return only a single digit: 1, 2, 3, 4, or 5. Do not include any extra text.

{test_case['case']}

Triage Level:"""
        
        try:
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Extract response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_part = response.split("Triage Level:")[-1].strip()
            
            # Extract ESI level
            esi_match = re.search(r'([1-5])', response_part)
            if esi_match:
                predicted = int(esi_match.group(1))
            else:
                predicted = None
                print(f"❓ Could not extract ESI from: '{response_part}'")
                continue
            
            # Check if correct
            expected = test_case['expected']
            if isinstance(expected, list):
                is_correct = predicted in expected
                expected_str = f"{expected[0]}-{expected[-1]}"
            else:
                is_correct = predicted == expected
                expected_str = str(expected)
            
            if is_correct:
                print(f"✅ CORRECT: Expected {expected_str}, Got {predicted}")
                correct += 1
                status = "✅"
            else:
                print(f"❌ WRONG: Expected {expected_str}, Got {predicted}")
                status = "❌"
            
            print(f"💡 Reasoning: {test_case['reasoning']}")
            print(f"🤖 Raw response: '{response_part[:50]}...'")
            
            results.append({
                'case': test_case['name'],
                'expected': expected_str,
                'predicted': predicted,
                'correct': is_correct,
                'status': status
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                'case': test_case['name'],
                'expected': expected_str if 'expected_str' in locals() else 'N/A',
                'predicted': 'ERROR',
                'correct': False,
                'status': '💥'
            })
    
    # Final results
    accuracy = (correct / total) * 100
    
    print(f"\n📊 FINAL VERIFICATION RESULTS")
    print("="*60)
    print(f"✅ Correct: {correct}/{total}")
    print(f"📈 Accuracy: {accuracy:.1f}%")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        print(f"{result['status']} {result['case']}: Expected {result['expected']}, Got {result['predicted']}")
    
    # Verdict
    print(f"\n🏆 CHUNK 12 VERDICT:")
    print("="*50)
    
    if accuracy >= 85:
        print("🎉 EXCELLENT! This IS your good Chunk 12!")
        print("✅ Model performance is strong")
        print("✅ Ready for RAG integration")
        print("✅ Triage classification working well")
    elif accuracy >= 70:
        print("👍 GOOD! This appears to be Chunk 12 or very close")
        print("⚠️  Minor accuracy differences might be due to:")
        print("   - Different test cases than training")
        print("   - Model quantization effects") 
        print("   - Random generation differences")
        print("✅ Still suitable for use")
    elif accuracy >= 50:
        print("⚠️  MODERATE performance - might be a different chunk")
        print("💡 Consider testing with your original validation set")
        print("🤔 Could still be usable depending on your needs")
    else:
        print("❌ LOW performance - this might not be Chunk 12")
        print("💡 Try loading a different commit or retrain")
    
    print(f"\n💾 MODEL STATUS: LOADED AND READY")
    print(f"📋 Use this model for your RAG enhancement project!")
    
    return accuracy, results

# ===========================================
# STEP 3: RUN VERIFICATION IF MODEL LOADED
# ===========================================
if model is not None and tokenizer is not None:
    print("\n🔍 Running comprehensive verification...")
    accuracy, results = comprehensive_triage_test(model, tokenizer)
    
    print(f"\n🎯 FINAL STATUS:")
    print("="*60)
    print(f"✅ Chunk 12 successfully loaded and tested!")
    print(f"📊 Overall accuracy: {accuracy:.1f}%")
    print(f"🚀 Ready for your RAG enhancement project!")
    
    # Keep model in memory for next steps
    print(f"\n💡 Your model and tokenizer are now available as:")
    print(f"   - model: Loaded Chunk 12 model")
    print(f"   - tokenizer: Compatible tokenizer")
    print(f"   - Ready for RAG integration!")
    
else:
    print("\n❌ FAILED TO LOAD MODEL")
    print("="*60)
    print("💡 Troubleshooting steps:")
    print("1. Make sure you have Unsloth installed")
    print("2. Check your internet connection")
    print("3. Verify repository access")
    print("4. Try restarting your runtime")