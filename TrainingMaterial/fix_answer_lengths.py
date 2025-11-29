import re

# Read the file
with open('multiple_choice.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Parse questions
lines = content.split('\n')
replacements = []

pattern = r'^\s*\("([^"]+)","([^"]+)","([^"]+)","([^"]+)","([^"]+)","([^"]+)","([^"]+)","([^"]+)"\)'

for i, line in enumerate(lines):
    match = re.match(pattern, line)
    if match:
        question, a1, a2, a3, a4, correct, category, source = match.groups()
        
        # Skip exam questions
        if 'eksamen' in source.lower():
            continue
        
        # Skip lab questions (they're code examples)
        if source.startswith('Lab'):
            continue
        
        correct_len = len(correct)
        min_len = correct_len * 0.9
        
        answers = [a1, a2, a3, a4]
        needs_fix = False
        
        for ans in answers:
            if ans != correct and len(ans) < min_len:
                needs_fix = True
                break
        
        if needs_fix:
            print(f"\nLine {i+1}: {question}")
            print(f"  Correct ({correct_len} chars): {correct}")
            print(f"  Source: {source}")
            
            # Create better wrong answers
            new_answers = []
            for ans in answers:
                if ans == correct:
                    new_answers.append(ans)
                elif len(ans) < min_len:
                    # Try to extend the answer to be similar length
                    target_len = int(correct_len * 0.95)  # 95% of correct answer length
                    diff = target_len - len(ans)
                    
                    # Add descriptive words to make it longer
                    if diff > 0:
                        if 'for' in correct.lower() or 'to' in correct.lower():
                            extensions = [' for improved results', ' in most cases', ' across different scenarios', 
                                        ' in typical situations', ' for better outcomes', ' under normal conditions',
                                        ' in standard applications', ' for optimal performance']
                        else:
                            extensions = [' and related concepts', ' in various contexts', ' across multiple domains',
                                        ' with specific characteristics', ' under certain conditions', ' in particular cases',
                                        ' with additional features', ' in specialized scenarios']
                        
                        # Find best extension
                        for ext in extensions:
                            if len(ans + ext) >= min_len and len(ans + ext) <= correct_len * 1.1:
                                new_ans = ans + ext
                                print(f"    Fixed: '{ans}' ({len(ans)}) -> '{new_ans}' ({len(new_ans)})")
                                new_answers.append(new_ans)
                                break
                        else:
                            new_answers.append(ans)  # Keep original if no good extension found
                    else:
                        new_answers.append(ans)
                else:
                    new_answers.append(ans)
            
            # Only add to replacements if we actually made changes
            if new_answers != answers:
                old_line = line
                new_line = f' ("{question}","{new_answers[0]}","{new_answers[1]}","{new_answers[2]}","{new_answers[3]}","{correct}","{category}","{source}"),'
                replacements.append((old_line, new_line))

print(f"\n\nTotal questions needing fixes: {len(replacements)}")
print("\nNote: These are automated suggestions. Please review and adjust manually for quality.")
print("Many answers may need custom rewording to maintain accuracy and clarity.")
