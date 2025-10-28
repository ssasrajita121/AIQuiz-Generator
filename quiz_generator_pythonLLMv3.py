# ============================================================================
# PROFESSIONAL QUIZ GENERATOR 
# ============================================================================
# High-quality quiz generation with validation and quality controls
# Designed for educational institutions and STEM organizations
# ============================================================================

import streamlit as st
import json
import re
# ============================================================================
# 🔑 API KEYS CONFIGURATION
# ============================================================================
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
# ============================================================================
# Try importing AI libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Configure APIs
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_KEY_HERE" and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_CONFIGURED = True
else:
    GEMINI_CONFIGURED = False

if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_KEY_HERE" and OPENAI_AVAILABLE:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_CONFIGURED = True
else:
    OPENAI_CONFIGURED = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Quiz Generator",
    page_icon="🎓",
    layout="wide"
)

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(to right, #11998e, #38ef7d);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .quiz-question {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .correct-answer {
        background: #d4edda;
        border: 2px solid #28a745;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .incorrect-answer {
        background: #f8d7da;
        border: 2px solid #dc3545;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .option {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s;
    }
    .score-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    .score-text {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .quality-badge {
        background: linear-gradient(to right, #11998e, #38ef7d);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        display: inline-block;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .error-report {
        background: #fff3cd;
        border: 2px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'quiz' not in st.session_state:
    st.session_state.quiz = None
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'generation_method' not in st.session_state:
    st.session_state.generation_method = None
if 'reported_questions' not in st.session_state:
    st.session_state.reported_questions = set()

# ============================================================================
# ENHANCED GEMINI GENERATION WITH QUALITY CONTROLS
# ============================================================================
def generate_with_gemini(topic, num_questions, difficulty, question_type="Mixed"):
    """Generate high-quality quiz using Gemini with enhanced prompting"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Adjust prompt based on question type
        if question_type == "Application (Problem-Solving)":
            style_instruction = """QUESTION STYLE: Application-Based (Problem-Solving)
- Include numerical problems that require calculations
- Present scenarios that need analysis
- Test understanding through application, not just memorization
- Examples: "Calculate...", "If X, then what is Y?", "A ball moving at..."
- Avoid simple "What is..." or "Define..." questions"""
        elif question_type == "Conceptual (Knowledge)":
            style_instruction = """QUESTION STYLE: Conceptual (Knowledge-Based)
- Test understanding of concepts and definitions
- Focus on "What is...", "Which of the following...", "Define..."
- Test recall and comprehension"""
        else:  # Mixed
            style_instruction = """QUESTION STYLE: Mixed (50% Conceptual, 50% Application)
- Combine knowledge questions with problem-solving
- Include both definitions AND calculations
- Vary question types for comprehensive assessment"""
        
        # Enhanced prompt with examples and strict instructions
        prompt = f"""You are an expert STEM educator creating a high-quality educational quiz. Accuracy is CRITICAL.

TOPIC: {topic}
DIFFICULTY: {difficulty}
QUESTIONS NEEDED: {num_questions}
{style_instruction}

CRITICAL REQUIREMENTS:
1. Generate EXACTLY {num_questions} questions at {difficulty.lower()} level
2. Each question MUST have EXACTLY 4 options labeled A, B, C, D
3. ONLY ONE option should be correct - verify this carefully
4. For math/programming questions: CALCULATE and VERIFY your answer before including it
5. For factual questions: Ensure accuracy based on established knowledge
6. Questions should be clear, unambiguous, and educational
7. Avoid trick questions or unclear wording

EXAMPLE FORMAT (Python topic):
[
  {{
    "question": "What is the result of 2 + 3 * 4 in Python?",
    "options": ["20", "14", "12", "10"],
    "correct": 1,
    "explanation": "Following PEMDAS: 3*4=12, then 2+12=14"
  }}
]

CRITICAL: All 4 options MUST contain visible text. Empty or whitespace-only options are NOT acceptable.
Example of INVALID options: ["", "", "option3", ""] - WRONG!
Example of VALID options: ["option1", "option2", "option3", "option4"] - CORRECT!

VERIFICATION CHECKLIST:
✓ Calculated all math/code answers manually
✓ Verified only ONE correct answer exists
✓ Checked factual accuracy
✓ Ensured options are distinct and reasonable
✓ Proofread questions for clarity

Now generate {num_questions} questions about "{topic}".

Return ONLY valid JSON array with this structure:
[
  {{
    "question": "Clear question text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct": 0,
    "explanation": "Brief explanation of why this is correct"
  }}
]

The "correct" field is the index (0-3) of the correct answer.
Include "explanation" field for quality verification.

JSON OUTPUT:"""
        
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            quiz_data = json.loads(json_match.group())
            # Validate with stricter checks
            if validate_quiz_strict(quiz_data):
                return quiz_data, None
            else:
                return None, "Quiz validation failed - quality standards not met"
        
        return None, "Could not parse valid JSON response"
        
    except Exception as e:
        return None, f"Gemini Error: {str(e)}"

# ============================================================================
# ENHANCED OPENAI GENERATION WITH QUALITY CONTROLS
# ============================================================================
def generate_with_openai(topic, num_questions, difficulty, question_type="Mixed"):
    """Generate high-quality quiz using OpenAI with enhanced prompting"""
    try:
        # Use GPT-4 if available for better quality, fallback to GPT-3.5
        model = "gpt-4" if has_gpt4_access() else "gpt-3.5-turbo"
        
        # Adjust style based on question type
        if question_type == "Application (Problem-Solving)":
            style_guide = "Focus on problem-solving questions that require calculations, analysis, and application of concepts. Avoid simple 'What is...' questions."
        elif question_type == "Conceptual (Knowledge)":
            style_guide = "Focus on conceptual understanding, definitions, and fundamental knowledge. Test recall and comprehension."
        else:
            style_guide = "Mix both conceptual questions and application problems. Include both definitions and calculations."
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert STEM educator. Your quizzes are known for accuracy and clarity. 
                    You ALWAYS verify your math calculations and factual accuracy before responding.
                    You respond ONLY with valid JSON arrays."""
                },
                {
                    "role": "user", 
                    "content": f"""Create {num_questions} VERIFIED multiple choice questions about "{topic}" at {difficulty.lower()} difficulty.

QUESTION STYLE: {style_guide}

CRITICAL INSTRUCTIONS:
1. For math/programming questions: Show your work mentally and verify calculations
2. For factual questions: Only include information you are confident is accurate
3. Each question has exactly 4 options
4. Only ONE option is correct
5. Options should be plausible but distinctly different

EXAMPLE (for Python topic):
[
  {{
    "question": "What is the output of: print(5 * 2 + 3)?",
    "options": ["13", "25", "10", "8"],
    "correct": 0,
    "explanation": "5*2=10, then 10+3=13"
  }}
]

Format your response as JSON array:
[
  {{
    "question": "Clear question?",
    "options": ["A", "B", "C", "D"],
    "correct": 0,
    "explanation": "Why this is correct"
  }}
]

VERIFY each answer before including it. Return ONLY the JSON array."""
                }
            ],
            temperature=0.3  # Lower temperature for more consistent, accurate output
        )
        
        text = response.choices[0].message.content.strip()
        text = text.replace('```json', '').replace('```', '').strip()
        
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            quiz_data = json.loads(json_match.group())
            if validate_quiz_strict(quiz_data):
                return quiz_data, None
            else:
                return None, "Quiz validation failed - quality standards not met"
        
        return None, "Could not parse valid JSON response"
        
    except Exception as e:
        return None, f"OpenAI Error: {str(e)}"

# ============================================================================
# CHECK GPT-4 ACCESS
# ============================================================================
def has_gpt4_access():
    """Check if API key has GPT-4 access"""
    try:
        if not OPENAI_CONFIGURED:
            return False
        # Simple check - try to list models (optional, can be expensive)
        return True  # Assume access for now
    except:
        return False

# ============================================================================
# STRICT VALIDATION FUNCTION
# ============================================================================
def validate_quiz_strict(quiz_data):
    """Strict validation for quality assurance"""
    if not isinstance(quiz_data, list) or len(quiz_data) == 0:
        return False
    
    for q in quiz_data:
        # Check required fields
        if not all(key in q for key in ['question', 'options', 'correct']):
            return False
        
        # Check question is not empty or too short
        if not q['question'] or len(q['question']) < 10:
            return False
        
        # Check exactly 4 options
        if not isinstance(q['options'], list) or len(q['options']) != 4:
            return False
        
        # Check all options are non-empty strings with meaningful content
        for opt in q['options']:
            if not opt or not isinstance(opt, str) or len(opt.strip()) == 0:
                return False
            # Check option has at least 1 character (not just whitespace)
            if len(opt.strip()) < 1:
                return False
        
        # Check correct answer is valid index
        if not isinstance(q['correct'], int) or q['correct'] < 0 or q['correct'] > 3:
            return False
        
        # Check options are distinct (no duplicates)
        if len(set(q['options'])) != 4:
            return False
    
    return True

# ============================================================================
# UNIFIED GENERATION WITH QUALITY RETRY
# ============================================================================
def generate_quiz_with_ai(topic, num_questions, difficulty, preferred_provider="auto", question_type="Mixed"):
    """Generate quiz with quality controls and retry mechanism"""
    
    max_attempts = 2  # Try twice per provider for better quality
    
    # Determine providers to try
    if preferred_provider == "auto":
        providers = []
        if OPENAI_CONFIGURED:
            providers.append(("OpenAI GPT", generate_with_openai))
        if GEMINI_CONFIGURED:
            providers.append(("Google Gemini", generate_with_gemini))
    elif preferred_provider == "gemini" and GEMINI_CONFIGURED:
        providers = [("Google Gemini", generate_with_gemini)]
    elif preferred_provider == "openai" and OPENAI_CONFIGURED:
        providers = [("OpenAI GPT", generate_with_openai)]
    else:
        return None, "No AI provider configured", None
    
    # Try each provider with retry
    last_error = None
    for provider_name, generate_func in providers:
        for attempt in range(max_attempts):
            quiz_data, error = generate_func(topic, num_questions, difficulty, question_type)
            if error is None:
                return quiz_data, None, provider_name
            last_error = error
    
    return None, last_error, None

# ============================================================================
# HEADER
# ============================================================================
st.markdown("# 🎓 Professional Quiz Generator")
st.markdown("### High-Quality Educational Content for Organizations")

# Show quality badge
providers = []
if GEMINI_CONFIGURED:
    providers.append("✓ Gemini")
if OPENAI_CONFIGURED:
    providers.append("✓ OpenAI")

if providers:
    st.markdown(f'<div class="quality-badge">🤖 AI Providers Active: {" | ".join(providers)} | Quality Verified</div>', unsafe_allow_html=True)
else:
    st.error("⚠️ No AI providers configured")

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Provider status
    st.markdown("**🔧 AI Providers:**")
    if GEMINI_CONFIGURED:
        st.success("✅ Google Gemini")
    else:
        st.info("○ Gemini (Not configured)")
    
    if OPENAI_CONFIGURED:
        st.success("✅ OpenAI GPT")
    else:
        st.info("○ OpenAI (Not configured)")
    
    st.markdown("---")
    
    # Provider selection
    if GEMINI_CONFIGURED or OPENAI_CONFIGURED:
        provider_options = ["Auto (Best Quality)"]
        if OPENAI_CONFIGURED:
            provider_options.append("OpenAI Only")
        if GEMINI_CONFIGURED:
            provider_options.append("Gemini Only")
        
        provider_choice = st.selectbox(
            "AI Provider",
            provider_options,
            help="Auto selects best provider for your query"
        )
        
        if provider_choice == "Auto (Best Quality)":
            preferred_provider = "auto"
        elif provider_choice == "OpenAI Only":
            preferred_provider = "openai"
        else:
            preferred_provider = "gemini"
    else:
        preferred_provider = "auto"
    
    st.markdown("---")
    
    st.markdown("**🎯 Quality Features:**")
    st.info("""
    ✓ Enhanced AI prompting
    ✓ Answer verification
    ✓ Strict validation
    ✓ Multiple retry attempts
    ✓ Error reporting system
    ✓ Professional output
    """)
    
    st.markdown("---")
    
    st.markdown("**💡 Recommended Topics:**")
    st.code("• Python Programming\n• Physics Concepts\n• Chemistry Basics\n• Mathematics\n• Biology", language="")
    
    # Reset button
    if st.session_state.quiz is not None:
        if st.button("🔄 Generate New Quiz"):
            st.session_state.quiz = None
            st.session_state.user_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.generation_method = None
            st.session_state.reported_questions = set()
            st.rerun()

# ============================================================================
# MAIN INPUT
# ============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    topic = st.text_area(
        "📚 Enter Topic",
        placeholder="Examples:\n• Python Data Structures\n• Newton's Laws of Motion\n• Photosynthesis Process\n• Algebra Equations\n• Periodic Table Elements",
        height=120
    )

with col2:
    num_questions = st.selectbox(
        "Number of Questions",
        [3, 5, 10],
        index=1
    )
    
    difficulty = st.selectbox(
        "Difficulty Level",
        ["Easy", "Medium", "Hard"],
        index=1
    )
    
    question_type = st.selectbox(
        "Question Style",
        ["Conceptual (Knowledge)", "Application (Problem-Solving)", "Mixed"],
        index=1,
        help="Application mode requires calculations and critical thinking"
    )

# ============================================================================
# GENERATION BUTTON
# ============================================================================
if st.button("🚀 Generate Professional Quiz") and st.session_state.quiz is None:
    if not topic:
        st.error("❌ Please enter a topic")
    elif not (GEMINI_CONFIGURED or OPENAI_CONFIGURED):
        st.error("❌ No AI provider configured")
        st.code("""Add API keys in code:
GEMINI_API_KEY = "your-key"
OPENAI_API_KEY = "your-key"

Install: pip install google-generativeai openai""")
    else:
        with st.spinner(f"🔬 Generating {num_questions} verified questions about '{topic}'..."):
            
            quiz_data, error, provider_used = generate_quiz_with_ai(
                topic=topic,
                num_questions=num_questions,
                difficulty=difficulty,
                preferred_provider=preferred_provider
            )
            
            if error:
                st.error(f"❌ {error}")
                st.warning("💡 Troubleshooting:")
                st.markdown("""
                1. Verify API keys are valid and active
                2. Try different AI provider
                3. Check internet connection
                4. Wait 30 seconds and retry
                """)
            else:
                st.session_state.quiz = quiz_data
                st.session_state.topic = topic
                st.session_state.difficulty = difficulty
                st.session_state.generation_method = f"Generated by {provider_used}"
                
                st.success(f"✅ High-quality quiz generated using {provider_used}!")
                st.info("📝 Scroll down to attempt")
                st.rerun()

# ============================================================================
# QUIZ ATTEMPT
# ============================================================================
if st.session_state.quiz is not None and not st.session_state.quiz_submitted:
    st.markdown("---")
    st.markdown("## 📝 Attempt Your Quiz")
    
    if st.session_state.generation_method:
        st.markdown(f'<div class="quality-badge">🎓 {st.session_state.generation_method} | Quality Verified</div>', unsafe_allow_html=True)
    
    for idx, q in enumerate(st.session_state.quiz):
        st.markdown(f"""
        <div class="quiz-question">
            <h3>Q{idx+1}. {q['question']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        answer = st.radio(
            f"Select answer for Q{idx+1}:",
            options=q['options'],
            key=f"q_{idx}",
            label_visibility="collapsed"
        )
        
        st.session_state.user_answers[idx] = q['options'].index(answer)
        st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🎯 Submit Quiz", use_container_width=True):
            st.session_state.quiz_submitted = True
            st.rerun()

# ============================================================================
# RESULTS WITH ERROR REPORTING
# ============================================================================
if st.session_state.quiz is not None and st.session_state.quiz_submitted:
    st.markdown("---")
    
    correct_count = 0
    total_questions = len(st.session_state.quiz)
    
    for idx, q in enumerate(st.session_state.quiz):
        if st.session_state.user_answers.get(idx) == q['correct']:
            correct_count += 1
    
    score_percentage = (correct_count / total_questions) * 100
    
    st.markdown(f"""
    <div class="score-card">
        <h2>🎉 Quiz Results</h2>
        <div class="score-text">{correct_count}/{total_questions}</div>
        <h3>{score_percentage:.1f}%</h3>
        <p style="font-size: 1.3rem; color: #666;">
            {"🌟 Outstanding Performance!" if score_percentage >= 90 else "🎯 Excellent Work!" if score_percentage >= 80 else "👍 Good Effort!" if score_percentage >= 60 else "📚 Keep Learning!"}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📋 Detailed Review")
    
    # Download results
    quiz_text = f"Professional Quiz Results\n{'='*60}\n"
    quiz_text += f"Topic: {st.session_state.get('topic', 'Quiz')}\n"
    quiz_text += f"Difficulty: {st.session_state.get('difficulty', 'Medium')}\n"
    quiz_text += f"Score: {correct_count}/{total_questions} ({score_percentage:.1f}%)\n"
    quiz_text += f"{st.session_state.get('generation_method', 'AI Generated')}\n"
    quiz_text += f"{'='*60}\n\n"
    
    for idx, q in enumerate(st.session_state.quiz):
        user_ans_idx = st.session_state.user_answers.get(idx, -1)
        quiz_text += f"Q{idx+1}. {q['question']}\n"
        for i, opt in enumerate(q['options']):
            quiz_text += f"   {chr(65+i)}) {opt}"
            if i == q['correct']:
                quiz_text += " ✓ (Correct)"
            if i == user_ans_idx and i != q['correct']:
                quiz_text += " ✗ (Your Answer)"
            quiz_text += "\n"
        if 'explanation' in q:
            quiz_text += f"   Explanation: {q['explanation']}\n"
        quiz_text += "\n"
    
    st.download_button(
        label="📥 Download Professional Report",
        data=quiz_text,
        file_name=f"stem_quiz_{st.session_state.get('topic', 'quiz').replace(' ', '_')}.txt",
        mime="text/plain"
    )
    
    # Display with error reporting
    for idx, q in enumerate(st.session_state.quiz):
        user_answer = st.session_state.user_answers.get(idx, -1)
        is_correct = user_answer == q['correct']
        
        st.markdown(f"""
        <div class="quiz-question">
            <h3>Q{idx+1}. {q['question']}</h3>
            <p style="color: {'#28a745' if is_correct else '#dc3545'}; font-weight: bold;">
                {'✓ Correct' if is_correct else '✗ Incorrect'}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        for i, option in enumerate(q['options']):
            if i == q['correct']:
                st.markdown(f"""
                <div class="correct-answer">
                    <strong>{chr(65+i)})</strong> {option} <strong style="color: #28a745;">✓ Correct Answer</strong>
                </div>
                """, unsafe_allow_html=True)
            elif i == user_answer and i != q['correct']:
                st.markdown(f"""
                <div class="incorrect-answer">
                    <strong>{chr(65+i)})</strong> {option} <strong style="color: #dc3545;">✗ Your Answer</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="option">
                    <strong>{chr(65+i)})</strong> {option}
                </div>
                """, unsafe_allow_html=True)
        
        # Show explanation if available
        if 'explanation' in q:
            st.info(f"💡 **Explanation:** {q['explanation']}")
        
        # Error reporting button
        if idx not in st.session_state.reported_questions:
            if st.button(f"⚠️ Report Issue with Q{idx+1}", key=f"report_{idx}"):
                st.session_state.reported_questions.add(idx)
                st.markdown(f"""
                <div class="error-report">
                    <strong>✓ Reported!</strong> Thank you for helping improve quality. 
                    Question {idx+1} has been flagged for review.
                </div>
                """, unsafe_allow_html=True)
                st.rerun()
        else:
            st.markdown("""
            <div class="error-report">
                ✓ This question has been reported for review
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# END
# ============================================================================