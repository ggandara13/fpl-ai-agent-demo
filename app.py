import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import random
import anthropic
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

# Page config MUST be first
st.set_page_config(page_title="FPL AI Agent - Multi-Level Demo", page_icon="⚡", layout="wide")

# Get API key from environment variable first, then allow override
env_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize session state for API key
if 'api_key' not in st.session_state:
    st.session_state.api_key = env_api_key

# Define structured output schemas
class ActionType(str, Enum):
    CREATE_TICKET = "create_ticket"
    SEND_NOTIFICATION = "send_notification"
    SCHEDULE_CALLBACK = "schedule_callback"
    APPLY_CREDIT = "apply_credit"
    EXTEND_PAYMENT = "extend_payment"
    ESCALATE = "escalate"
    UPDATE_ACCOUNT = "update_account"
    PRIORITY_RESTORATION = "priority_restoration"

class Action(BaseModel):
    type: ActionType
    priority: str = Field(description="high, medium, or low")
    description: str
    data: Dict[str, Any] = Field(default_factory=dict)
    automated: bool = Field(default=True, description="Should this action be automated?")
    completed: bool = Field(default=False)
    timestamp: Optional[str] = None

class StructuredResponse(BaseModel):
    """Structured output from the AI agent"""
    # Customer-facing response
    customer_response: str = Field(description="The message to show to the customer")
    
    # Intent analysis
    intent: str = Field(description="Classified intent: outage_report, billing_inquiry, etc.")
    intent_confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    
    # Extracted entities
    entities: Dict[str, Any] = Field(default_factory=dict)
    
    # Sentiment analysis
    sentiment_score: float = Field(ge=-1, le=1)
    emotional_state: str = Field(default="neutral")
    
    # Recommended actions
    actions: List[Action] = Field(default_factory=list)
    
    # Next steps in conversation
    next_steps: List[str] = Field(default_factory=list)
    
    # Internal notes
    internal_notes: str = Field(default="")
    
    # Escalation decision
    needs_human: bool = Field(default=False)
    escalation_reason: Optional[str] = Field(default=None)

# Sidebar for API Key configuration
with st.sidebar:
    st.markdown("### 🔑 Claude API Configuration")
    
    if env_api_key:
        st.success(f"✅ API Key loaded from environment")
        use_env_key = st.checkbox("Use environment API key", value=True)
        if not use_env_key:
            api_key_input = st.text_input(
                "Override with different API Key", 
                value="",
                type="password",
                help="Enter a different API key to override the environment variable"
            )
            if api_key_input:
                st.session_state.api_key = api_key_input
        else:
            st.session_state.api_key = env_api_key
    else:
        api_key_input = st.text_input(
            "Enter Claude API Key", 
            value=st.session_state.api_key or "",
            type="password",
            help="Required for Claude API level. Get your key from console.anthropic.com"
        )
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ No API key - Claude level will use demo mode")
    
    # Test API button
    if st.button("🧪 Test Claude API"):
        if st.session_state.api_key:
            try:
                client = anthropic.Anthropic(api_key=st.session_state.api_key)
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=50,
                    messages=[{"role": "user", "content": "Say 'API works!'"}]
                )
                st.success(f"✅ API Test Success: {response.content[0].text}")
            except Exception as e:
                st.error(f"❌ API Test Failed: {str(e)}")
        else:
            st.error("No API key provided")
    
    # Debug info
    with st.expander("🐛 Debug Info"):
        st.write(f"API Key exists: {bool(st.session_state.get('api_key'))}")
        st.write(f"API Key length: {len(st.session_state.get('api_key', ''))}")
        if 'agent' in st.session_state:
            st.write(f"Agent has_claude_api: {st.session_state.agent.has_claude_api}")
            st.write(f"Agent claude_client: {st.session_state.agent.claude_client is not None}")
    
    st.markdown("---")

class CustomerIntent(Enum):
    BILLING_INQUIRY = "billing_inquiry"
    OUTAGE_REPORT = "outage_report"
    SERVICE_REQUEST = "service_request"
    PAYMENT = "payment"
    GENERAL_INQUIRY = "general_inquiry"
    COMPLAINT = "complaint"

@dataclass
class ProcessingStep:
    name: str
    description: str
    result: any
    confidence: float
    processing_time: float

class AILevel(Enum):
    BASIC = "Basic NLP (Keywords)"
    MEDIUM = "Advanced NLP (ML Models)"
    ADVANCED = "Local LLM (Llama/Mistral)"
    PREMIUM = "Claude API (Most Intelligent)"

@dataclass
class Memory:
    """AI Agent Memory System - One of the 3 core pillars"""
    # Short-term memory (current conversation)
    conversation_history: List[Dict] = field(default_factory=list)
    current_intent: Optional[CustomerIntent] = None
    current_entities: Dict = field(default_factory=dict)
    
    # Long-term memory (customer profile)
    customer_id: str = "12345"
    account_number: str = "1234567890"
    location: str = "Miami Beach"
    service_history: List[Dict] = field(default_factory=list)
    payment_history: List[Dict] = field(default_factory=lambda: [
        {"amount": 125.50, "due_date": (datetime.now() + timedelta(days=2)).isoformat(), "status": "pending"}
    ])
    preference_profile: Dict = field(default_factory=lambda: {"language": "en", "contact_method": "text"})
    
    # Episodic memory (past interactions)
    past_interactions: List[Dict] = field(default_factory=lambda: [
        {"date": "2024-12-15", "issue": "Brief power outage", "resolved": True},
        {"date": "2024-11-20", "issue": "High bill inquiry", "resolved": True}
    ])
    resolved_issues: List[Dict] = field(default_factory=list)
    sentiment_trend: List[float] = field(default_factory=list)
    
    # Working memory (current context)
    area_outages: Dict = field(default_factory=lambda: {
        "miami beach": {"active": True, "start_time": datetime.now() - timedelta(hours=2), "affected_customers": 2400},
        "coral gables": {"active": False}
    })
    weather_alerts: Dict = field(default_factory=lambda: {"storm_warning": False})
    system_status: Dict = field(default_factory=lambda: {"maintenance": False})
    
    # Action history
    executed_actions: List[Action] = field(default_factory=list)
    
    def remember_interaction(self, message: str, intent: CustomerIntent, sentiment: float):
        """Store interaction in memory"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "intent": intent,
            "sentiment": sentiment
        })
        self.sentiment_trend.append(sentiment)
        if len(self.sentiment_trend) > 10:
            self.sentiment_trend.pop(0)
    
    def log_action(self, action: Action):
        """Log executed action"""
        action.timestamp = datetime.now().isoformat()
        action.completed = True
        self.executed_actions.append(action)
    
    def get_customer_context(self) -> Dict:
        """Get comprehensive customer context"""
        return {
            "is_repeat_issue": len([i for i in self.past_interactions if "outage" in i.get("issue", "").lower()]) > 0,
            "sentiment_declining": len(self.sentiment_trend) > 2 and sum(self.sentiment_trend[-3:]) / 3 < -0.3,
            "high_value_customer": len(self.payment_history) > 12,
            "preferred_language": self.preference_profile.get("language", "en"),
            "past_satisfaction": 4.2,
            "account_status": "active",
            "payment_due_soon": True
        }

class Goal:
    """AI Agent Goal System - One of the 3 core pillars"""
    def __init__(self):
        self.primary_goals = {
            "resolve_issue": {"priority": 1, "metric": "first_contact_resolution"},
            "customer_satisfaction": {"priority": 2, "metric": "csat_score"},
            "prevent_escalation": {"priority": 3, "metric": "escalation_rate"}
        }
        
        self.active_goals = []
        self.completed_goals = []
    
    def set_goal_from_intent(self, intent: CustomerIntent, context: Dict):
        """Set specific goals based on intent and context"""
        self.active_goals = []
        
        if intent == CustomerIntent.OUTAGE_REPORT:
            self.active_goals.extend([
                {"goal": "confirm_outage_location", "status": "pending"},
                {"goal": "provide_restoration_time", "status": "pending"},
                {"goal": "offer_updates", "status": "pending"}
            ])
            if context.get("sentiment_declining"):
                self.active_goals.append({"goal": "prevent_complaint_escalation", "status": "active"})
        
        elif intent == CustomerIntent.PAYMENT:
            self.active_goals.extend([
                {"goal": "process_payment", "status": "pending"},
                {"goal": "suggest_autopay", "status": "pending"}
            ])
        
        return self.active_goals

class MultiLevelAgent:
    def __init__(self, api_key=None):
        self.current_level = AILevel.BASIC
        self.processing_steps = []
        
        # Three Core Pillars
        self.brain = None  # LLM/NLP system (changes based on level)
        self.memory = Memory()  # Persistent memory system
        self.goal = Goal()  # Goal-oriented behavior
        
        # Initialize Claude client if API key available
        self.claude_client = None
        self.has_claude_api = False
        
        if api_key and api_key != "your-api-key-here" and len(api_key) > 10:
            try:
                self.claude_client = anthropic.Anthropic(api_key=api_key)
                # Don't test the API here - just assume it works if we can create the client
                self.has_claude_api = True
                print(f"Claude API client created with key length: {len(api_key)}")
            except Exception as e:
                print(f"Claude API client creation failed: {e}")
                self.claude_client = None
                self.has_claude_api = False
        else:
            print(f"No valid API key provided. Key length: {len(api_key) if api_key else 0}")
    
    def get_pipeline_steps(self, ai_level: AILevel) -> List[Dict]:
        """Get the processing pipeline for each AI level"""
        pipelines = {
            AILevel.BASIC.value: [
                {"step": "Keyword Extraction", "icon": "🔤", "description": "Simple pattern matching"},
                {"step": "Regex Patterns", "icon": "📐", "description": "Account/location detection"},
                {"step": "Template Response", "icon": "📝", "description": "Pre-defined templates"},
                {"step": "Basic Memory", "icon": "💾", "description": "Store conversation"},
            ],
            AILevel.MEDIUM.value: [
                {"step": "spaCy NER", "icon": "🏷️", "description": "Named entity recognition"},
                {"step": "BERT Classifier", "icon": "🧠", "description": "Intent classification"},
                {"step": "Sentiment Analysis", "icon": "😊", "description": "DistilBERT emotions"},
                {"step": "Context Memory", "icon": "🗂️", "description": "Multi-turn context"},
                {"step": "Smart Templates", "icon": "💬", "description": "Dynamic responses"},
            ],
            AILevel.ADVANCED.value: [
                {"step": "Local LLM", "icon": "🤖", "description": "Llama/Mistral model"},
                {"step": "Semantic Understanding", "icon": "🎯", "description": "Deep comprehension"},
                {"step": "Memory Retrieval", "icon": "🔍", "description": "Similar case search"},
                {"step": "Goal Planning", "icon": "📋", "description": "Multi-step goals"},
                {"step": "Generated Response", "icon": "✨", "description": "Natural language"},
            ],
            AILevel.PREMIUM.value: [
                {"step": "Claude Analysis", "icon": "🌟", "description": "Advanced reasoning"},
                {"step": "Full Context", "icon": "🌐", "description": "Complete understanding"},
                {"step": "Predictive Memory", "icon": "🔮", "description": "Anticipate needs"},
                {"step": "Strategic Goals", "icon": "♟️", "description": "Long-term planning"},
                {"step": "Structured Output", "icon": "🎯", "description": "Actionable decisions"},
                {"step": "Automated Actions", "icon": "⚡", "description": "Execute decisions"},
            ]
        }
        return pipelines.get(ai_level.value, [])
    
    def process_message(self, message: str, ai_level: AILevel) -> Dict:
        """Process message with selected AI level - NOW WITH STRUCTURED OUTPUT"""
        self.current_level = ai_level
        self.processing_steps = []
        
        print(f"\n=== PROCESS MESSAGE DEBUG ===")
        print(f"ai_level parameter: {ai_level}")
        print(f"ai_level.value: {ai_level.value}")
        print(f"self.current_level: {self.current_level}")
        print(f"AILevel.MEDIUM.value: {AILevel.MEDIUM.value}")
        
        start_time = time.time()
        
        # ALWAYS use structured output for Premium level if Claude is available
        if ai_level.value == "Claude API (Most Intelligent)" and self.has_claude_api:
            print(">>> USING STRUCTURED OUTPUT FOR CLAUDE")
            try:
                return self._process_with_structured_output(message)
            except Exception as e:
                print(f">>> STRUCTURED OUTPUT ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                # Don't fall back - show the error
                return {
                    "response": f"Error using structured output: {str(e)}",
                    "intent": "error",
                    "intent_confidence": 0,
                    "sentiment": 0,
                    "entities": {},
                    "proactive_actions": [],
                    "processing_steps": [ProcessingStep(
                        name="Error",
                        description=str(e),
                        result={},
                        confidence=0,
                        processing_time=0
                    )],
                    "processing_time": time.time() - start_time,
                    "ai_level": ai_level.value,
                    "error": str(e)
                }
        else:
            # Legacy processing for other levels
            return self._process_legacy(message, ai_level)
    
    def _process_with_structured_output(self, message: str) -> Dict:
        """Process using structured output - the modern way"""
        start_time = time.time()
        
        try:
            # Build comprehensive context
            context = self.memory.get_customer_context()
            area_outage = self.memory.area_outages.get("miami beach", {})
            
            # Create structured prompt for Claude
            system_prompt = """You are an expert FPL (Florida Power & Light) customer service AI agent.
            You must analyze the customer message and return a valid JSON response with this exact structure:
            
            {
                "customer_response": "Natural, helpful message for the customer",
                "intent": "one of: outage_report, billing_inquiry, payment, complaint, service_request, general_inquiry",
                "intent_confidence": 0.0 to 1.0,
                "entities": {
                    "location": "extracted location or null",
                    "duration": "how long the issue has been occurring",
                    "account_number": "extracted account or null",
                    "urgency": "high, medium, or low"
                },
                "sentiment_score": -1.0 to 1.0,
                "emotional_state": "frustrated, angry, calm, satisfied, etc.",
                "actions": [
                    {
                        "type": "one of: create_ticket, send_notification, apply_credit, extend_payment, escalate, priority_restoration",
                        "priority": "high, medium, or low",
                        "description": "What this action does",
                        "data": {"relevant": "data for the action"},
                        "automated": true or false
                    }
                ],
                "next_steps": ["list", "of", "suggested", "follow-ups"],
                "internal_notes": "Important observations for internal use",
                "needs_human": true or false,
                "escalation_reason": "Why human is needed (if applicable)"
            }
            
            Be empathetic and professional. If it's an outage, always mention the restoration time of 2:30 PM."""
            
            user_prompt = f"""Customer Message: "{message}"
            
Customer Context:
- ID: {self.memory.customer_id}
- Location: {self.memory.location}
- Account: {self.memory.account_number}
- Past interactions: {len(self.memory.past_interactions)}
- Sentiment trend: {'declining' if context['sentiment_declining'] else 'stable'}
- Area outage active: {area_outage.get('active', False)}
- Affected customers: {area_outage.get('affected_customers', 0)}
- Payment due: ${self.memory.payment_history[0]['amount']} in {(datetime.fromisoformat(self.memory.payment_history[0]['due_date']) - datetime.now()).days} days

Generate the JSON response following the exact structure provided."""

            # Call Claude for structured output
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.3,  # Lower for consistent structure
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Parse the JSON response
            json_str = response.content[0].text
            
            # Extract JSON if wrapped in markdown
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            # Parse JSON
            structured_data = json.loads(json_str)
            
            # Validate and create structured response
            structured_response = StructuredResponse(**structured_data)
            
            # Execute automated actions
            executed_actions = []
            for action in structured_response.actions:
                if action.automated:
                    self._execute_action(action)
                    executed_actions.append(action)
            
            # Log to memory
            if structured_response.intent in [e.value for e in CustomerIntent]:
                intent_enum = CustomerIntent(structured_response.intent)
                self.memory.remember_interaction(message, intent_enum, structured_response.sentiment_score)
            
            # Create response with all structured data
            total_time = time.time() - start_time
            
            return {
                "response": structured_response.customer_response,
                "intent": structured_response.intent,
                "intent_confidence": structured_response.intent_confidence,
                "sentiment": structured_response.sentiment_score,
                "entities": structured_response.entities,
                "proactive_actions": [
                    {
                        "type": action.type.value,
                        "message": action.description,
                        "priority": action.priority,
                        "automated": action.automated,
                        "completed": action in executed_actions
                    }
                    for action in structured_response.actions
                ],
                "processing_steps": [
                    ProcessingStep(
                        name="Structured Analysis",
                        description="Claude analyzed message and generated structured output",
                        result={"actions_generated": len(structured_response.actions)},
                        confidence=0.98,
                        processing_time=total_time
                    )
                ],
                "processing_time": total_time,
                "ai_level": AILevel.PREMIUM.value,
                "structured_output": structured_response.model_dump(),
                "next_steps": structured_response.next_steps,
                "needs_human": structured_response.needs_human,
                "escalation_reason": structured_response.escalation_reason,
                "emotional_state": structured_response.emotional_state,
                "internal_notes": structured_response.internal_notes,
                "executed_actions": [a.model_dump() for a in executed_actions]
            }
            
        except Exception as e:
            print(f"Structured output error: {str(e)}")
            # Fallback to legacy processing
            return self._process_legacy(message, AILevel.PREMIUM)
    
    def _execute_action(self, action: Action):
        """Execute an action - this is where the magic happens"""
        print(f"🎯 Executing action: {action.type.value} - {action.description}")
        
        if action.type == ActionType.CREATE_TICKET:
            ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            action.data["ticket_id"] = ticket_id
            print(f"   ✅ Created ticket: {ticket_id}")
            
        elif action.type == ActionType.SEND_NOTIFICATION:
            print(f"   ✅ Sending {action.data.get('channel', 'SMS')}: {action.data.get('message', action.description)}")
            
        elif action.type == ActionType.APPLY_CREDIT:
            amount = action.data.get('amount', 50)
            print(f"   ✅ Applied ${amount} credit to account {self.memory.account_number}")
            
        elif action.type == ActionType.EXTEND_PAYMENT:
            days = action.data.get('days', 7)
            print(f"   ✅ Extended payment due date by {days} days")
            
        elif action.type == ActionType.PRIORITY_RESTORATION:
            print(f"   ✅ Added to priority restoration list for {action.data.get('reason', 'critical customer')}")
            
        elif action.type == ActionType.SCHEDULE_CALLBACK:
            print(f"   ✅ Scheduled callback: {action.data.get('time', 'within 24 hours')}")
        
        # Log the action
        self.memory.log_action(action)
    
    def _process_legacy(self, message: str, ai_level: AILevel) -> Dict:
        """Legacy processing for non-structured output levels"""
        start_time = time.time()  # Add this line
        
        # Step 1: Entity Extraction
        entities = self._extract_entities(message)
        
        # Step 2: Intent Classification
        intent, intent_confidence = self._classify_intent(message)
        
        # Step 3: Sentiment Analysis
        sentiment = self._analyze_sentiment(message)
        
        # Step 4: Memory Retrieval
        memory_context = self._retrieve_memory_context(message, intent)
        
        # Step 5: Goal Setting
        goals = self._set_goals(intent, memory_context)
        
        # Step 6: Generate Response
        response = self._generate_response(message, intent, entities, memory_context, sentiment)
        
        # Step 7: Proactive Analysis
        proactive_actions = self._analyze_proactive_opportunities(message, intent, entities, memory_context, sentiment)
        
        # Update memory with this interaction
        self.memory.remember_interaction(message, intent, sentiment)
        
        total_time = time.time() - start_time
        
        return {
            "response": response,
            "intent": intent.value,
            "intent_confidence": intent_confidence,
            "sentiment": sentiment,
            "entities": entities,
            "proactive_actions": proactive_actions,
            "processing_steps": self.processing_steps,
            "processing_time": total_time,
            "ai_level": ai_level.value,
            "memory_context": memory_context,
            "active_goals": goals,
            "pipeline_used": self.get_pipeline_steps(ai_level)
        }
    
    def _extract_entities(self, message: str) -> Dict:
        """Extract entities based on AI level - SHOWS CLEAR DIFFERENCES"""
        start = time.time()
        entities = {"locations": [], "account_numbers": [], "temporal": []}
        
        if self.current_level == AILevel.BASIC:
            # BASIC: Only finds exact keyword matches
            import re
            
            # Only finds exact location names
            if "miami beach" in message.lower():
                entities["locations"].append("Miami Beach")
            
            # Basic account pattern
            account_pattern = r'\b\d{10}\b'
            entities["account_numbers"] = re.findall(account_pattern, message)
            
            # Misses temporal references
            confidence = 0.6
            
        elif self.current_level == AILevel.MEDIUM:
            # MEDIUM: Better entity recognition
            import re
            
            # Finds variations
            if any(word in message.lower() for word in ["miami", "beach", "home", "house"]):
                entities["locations"].append("Miami Beach" if "miami" in message.lower() else "Customer Residence")
            
            # Temporal extraction
            if "yesterday" in message.lower():
                entities["temporal"].append("yesterday")
            if "2 hours" in message.lower():
                entities["temporal"].append("2 hours ago")
            elif "hours" in message.lower():
                entities["temporal"].append("recent")
                
            confidence = 0.8
            
        elif self.current_level == AILevel.ADVANCED:
            # ADVANCED: Sophisticated extraction
            # Understands context
            entities["locations"].append("Miami Beach")  # Infers from memory
            
            # Better temporal understanding
            if "yesterday" in message.lower():
                entities["temporal"].append((datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
            elif "since" in message.lower():
                entities["temporal"].append("extended_duration")
                
            # Extracts implicit information
            if "work from home" in message.lower():
                entities["work_impact"] = True
                
            confidence = 0.9
            
        else:  # PREMIUM
            # PREMIUM: Full understanding
            entities["locations"].append("Miami Beach")
            
            # Precise temporal extraction
            if "yesterday" in message.lower():
                entities["temporal"].append({
                    "start": (datetime.now() - timedelta(days=1)).isoformat(),
                    "duration": "24+ hours",
                    "urgency": "high"
                })
            
            # Extracts all context
            entities["work_impact"] = "work from home" in message.lower()
            entities["emotional_state"] = "frustrated" if "unacceptable" in message.lower() else "concerned"
            entities["customer_location"] = self.memory.location
            
            confidence = 0.98
        
        self.processing_steps.append(ProcessingStep(
            name="Entity Extraction",
            description=f"Extracted entities using {self.current_level.value}",
            result=entities,
            confidence=confidence,
            processing_time=time.time() - start
        ))
        
        return entities
    
    def _classify_intent(self, message: str) -> Tuple[CustomerIntent, float]:
        """Classify intent - SHOWS CLEAR DIFFERENCES"""
        start = time.time()
        message_lower = message.lower()
        
        if self.current_level == AILevel.BASIC:
            # BASIC: Needs exact keywords
            if any(phrase in message_lower for phrase in ["power out", "no electricity", "outage"]):
                intent = CustomerIntent.OUTAGE_REPORT
                confidence = 0.8
            elif any(phrase in message_lower for phrase in ["no power", "blackout"]):
                intent = CustomerIntent.OUTAGE_REPORT
                confidence = 0.7
            else:
                # Fails on "I have not power" - no exact match
                intent = CustomerIntent.GENERAL_INQUIRY
                confidence = 0.3
                
        elif self.current_level == AILevel.MEDIUM:
            # MEDIUM: Better pattern recognition
            power_words = ["power", "electricity", "electric", "light", "outage"]
            negative_words = ["no", "not", "out", "off", "don't", "without"]
            
            has_power_word = any(word in message_lower for word in power_words)
            has_negative = any(word in message_lower for word in negative_words)
            
            if has_power_word and has_negative:
                intent = CustomerIntent.OUTAGE_REPORT
                confidence = 0.85
            elif has_power_word:
                intent = CustomerIntent.GENERAL_INQUIRY
                confidence = 0.6
            else:
                intent = CustomerIntent.GENERAL_INQUIRY
                confidence = 0.5
                
        elif self.current_level == AILevel.ADVANCED:
            # ADVANCED: Understands context and grammar
            # Can handle "I have not power" correctly
            if ("have not" in message_lower or "don't have" in message_lower or "no" in message_lower) and \
               any(word in message_lower for word in ["power", "electricity"]):
                intent = CustomerIntent.OUTAGE_REPORT
                confidence = 0.92
            elif "work from home" in message_lower and any(word in message_lower for word in ["off", "out"]):
                intent = CustomerIntent.COMPLAINT  # Understands urgency
                confidence = 0.88
            else:
                intent = CustomerIntent.GENERAL_INQUIRY
                confidence = 0.85
                
        else:  # PREMIUM
            # PREMIUM: Perfect understanding
            # Understands any variation including poor grammar
            if any(phrase in message_lower for phrase in ["have not power", "don't have power", "no power", 
                                                         "power out", "electricity off", "electrical outage"]):
                # Also considers emotional context
                if "unacceptable" in message_lower or "work from home" in message_lower:
                    intent = CustomerIntent.COMPLAINT  # Escalated due to impact
                else:
                    intent = CustomerIntent.OUTAGE_REPORT
                confidence = 0.98
            else:
                intent = CustomerIntent.GENERAL_INQUIRY
                confidence = 0.95
        
        self.processing_steps.append(ProcessingStep(
            name="Intent Classification",
            description=f"Classified intent using {self.current_level.value}",
            result={"intent": intent.value, "confidence": confidence},
            confidence=confidence,
            processing_time=time.time() - start
        ))
        
        self.memory.current_intent = intent
        return intent, confidence
    
    def _analyze_sentiment(self, message: str) -> float:
        """Analyze sentiment - SHOWS CLEAR DIFFERENCES"""
        start = time.time()
        
        if self.current_level == AILevel.BASIC:
            # BASIC: Simple word counting
            positive_words = ["thank", "great", "good"]
            negative_words = ["not", "no", "bad"]
            
            pos_count = sum(1 for word in positive_words if word in message.lower())
            neg_count = sum(1 for word in negative_words if word in message.lower())
            
            # Misses nuanced emotion
            sentiment = -0.2 if neg_count > pos_count else 0.0
            confidence = 0.5
            
        elif self.current_level == AILevel.MEDIUM:
            # MEDIUM: Better emotion detection
            if "unacceptable" in message.lower():
                sentiment = -0.8
            elif any(word in message.lower() for word in ["frustrated", "angry"]):
                sentiment = -0.7
            elif "not" in message.lower() or "no" in message.lower():
                sentiment = -0.4
            else:
                sentiment = 0.0
            confidence = 0.75
            
        elif self.current_level == AILevel.ADVANCED:
            # ADVANCED: Context-aware sentiment
            # Understands situational frustration
            if "work from home" in message.lower() and "off" in message.lower():
                sentiment = -0.7  # High impact = negative
            elif "yesterday" in message.lower():
                sentiment = -0.8  # Duration increases negativity
            elif "not power" in message.lower():
                sentiment = -0.5
            else:
                sentiment = -0.3
            confidence = 0.85
            
        else:  # PREMIUM
            # PREMIUM: Nuanced emotional understanding
            # Analyzes multiple factors
            factors = {
                "duration": -0.3 if "yesterday" in message.lower() or "hours" in message.lower() else 0,
                "impact": -0.3 if "work from home" in message.lower() else 0,
                "language": -0.2 if "unacceptable" in message.lower() else 0,
                "basic_frustration": -0.2
            }
            sentiment = max(-1.0, sum(factors.values()))
            confidence = 0.95
        
        self.processing_steps.append(ProcessingStep(
            name="Sentiment Analysis",
            description=f"Analyzed sentiment using {self.current_level.value}",
            result={"sentiment": sentiment, "factors": factors if self.current_level == AILevel.PREMIUM else None},
            confidence=confidence,
            processing_time=time.time() - start
        ))
        
        return sentiment
    
    def _retrieve_memory_context(self, message: str, intent: CustomerIntent) -> Dict:
        """Retrieve memory - SHOWS DIFFERENT DEPTH BY LEVEL"""
        start = time.time()
        
        print(f">>> _retrieve_memory_context called with level: {self.current_level}")
        
        if self.current_level == AILevel.BASIC:
            # BASIC: Minimal memory usage
            context = {
                "has_account": True,
                "location_known": False
            }
            
        elif self.current_level == AILevel.MEDIUM:
            # MEDIUM: Basic context retrieval
            # FIX: Extract location from entities to determine area outage
            location_key = "miami beach"  # Default
            if "miami" in message.lower() or "beach" in message.lower():
                location_key = "miami beach"
            elif "coral" in message.lower() or "gables" in message.lower():
                location_key = "coral gables"
            
            context = {
                "account_number": self.memory.account_number,
                "past_issues": len(self.memory.past_interactions),
                "area_status": "active" if self.memory.area_outages.get(location_key, {}).get("active", False) else "clear",
                "area_outage_active": self.memory.area_outages.get(location_key, {}).get("active", False)  # FIX: Add this key
            }
            
        elif self.current_level == AILevel.ADVANCED:
            # ADVANCED: Rich context
            context = {
                "account_number": self.memory.account_number,
                "location": self.memory.location,
                "past_outages": [i for i in self.memory.past_interactions if "outage" in i.get("issue", "").lower()],
                "area_outage_active": self.memory.area_outages.get("miami beach", {}).get("active", False),
                "payment_due_soon": True
            }
            
        else:  # PREMIUM
            # PREMIUM: Complete context with predictions
            customer_context = self.memory.get_customer_context()
            context = {
                **customer_context,
                "full_profile": {
                    "account": self.memory.account_number,
                    "location": self.memory.location,
                    "service_history": self.memory.past_interactions,
                    "satisfaction_score": 4.2,
                    "lifetime_value": "$1,506"
                },
                "area_outage": self.memory.area_outages.get("miami beach", {}),
                "area_outage_active": self.memory.area_outages.get("miami beach", {}).get("active", False),  # Add this for compatibility
                "predicted_needs": ["restoration_time", "compensation", "work_from_home_support"],
                "escalation_risk": "high" if "unacceptable" in message else "medium"
            }
        
        print(f">>> Context created for {self.current_level}: {list(context.keys())}")
        
        self.processing_steps.append(ProcessingStep(
            name="Memory Retrieval",
            description=f"Retrieved context using {self.current_level.value}",
            result=context,
            confidence=0.95,
            processing_time=time.time() - start
        ))
        
        return context
    
    def _set_goals(self, intent: CustomerIntent, context: Dict) -> List[Dict]:
        """Set goals - MORE SOPHISTICATED AT HIGHER LEVELS"""
        start = time.time()
        
        if self.current_level == AILevel.BASIC:
            # BASIC: Simple goal
            goals = [{"goal": "answer_question", "status": "active"}]
            
        elif self.current_level == AILevel.MEDIUM:
            # MEDIUM: Intent-based goals
            if intent == CustomerIntent.OUTAGE_REPORT:
                goals = [
                    {"goal": "confirm_outage", "status": "active"},
                    {"goal": "get_account_info", "status": "pending"}
                ]
            else:
                goals = [{"goal": "understand_request", "status": "active"}]
                
        elif self.current_level == AILevel.ADVANCED:
            # ADVANCED: Multi-step goals
            goals = self.goal.set_goal_from_intent(intent, context)
            if context.get("payment_due_soon"):
                goals.append({"goal": "offer_payment_extension", "status": "pending"})
                
        else:  # PREMIUM
            # PREMIUM: Strategic goals
            goals = self.goal.set_goal_from_intent(intent, context)
            
            # Add predictive goals
            if context.get("escalation_risk") == "high":
                goals.insert(0, {"goal": "de-escalate_situation", "status": "urgent"})
            
            if context.get("area_outage", {}).get("active"):
                goals.extend([
                    {"goal": "provide_precise_restoration", "status": "active"},
                    {"goal": "offer_compensation_options", "status": "pending"},
                    {"goal": "setup_restoration_alerts", "status": "pending"}
                ])
        
        self.processing_steps.append(ProcessingStep(
            name="Goal Setting",
            description=f"Set {len(goals)} goals using {self.current_level.value}",
            result={"goals": [g["goal"] for g in goals]},
            confidence=0.9,
            processing_time=time.time() - start
        ))
        
        return goals
    
    def _generate_response(self, message: str, intent: CustomerIntent, entities: Dict, context: Dict, sentiment: float) -> str:
        """Generate response - VERY DIFFERENT BY LEVEL"""
        start = time.time()
        
        print(f"\n=== GENERATE RESPONSE DEBUG ===")
        print(f"Current Level: {self.current_level}")
        print(f"Current Level name: {self.current_level.name}")
        print(f"Current Level value: {self.current_level.value}")
        print(f"Intent: {intent}")
        
        response = ""  # Initialize response
        
        # BASIC LEVEL
        if self.current_level.name == "BASIC":
            print(">>> Generating BASIC response")
            if intent == CustomerIntent.OUTAGE_REPORT:
                response = "Outage reported. Ticket created. We will send updates."
            elif intent == CustomerIntent.BILLING_INQUIRY:
                response = "For billing, provide account number."
            elif intent == CustomerIntent.COMPLAINT:
                response = "We apologize. Agent will contact you."
            elif intent == CustomerIntent.PAYMENT:
                response = "To pay bill, visit website or call 1-800-XXX-XXXX."
            elif intent == CustomerIntent.SERVICE_REQUEST:
                response = "Service request received. Reference number: SR-12345."
            else:
                response = "How can I help?"
                
        # MEDIUM LEVEL
        elif self.current_level.name == "MEDIUM":
            print(">>> Generating MEDIUM response")
            if intent == CustomerIntent.OUTAGE_REPORT:
                location = entities.get('locations', ['your area'])[0] if entities.get('locations') else "your area"
                response = f"Power outage confirmed in {location}. Multiple customers affected. Estimated restoration: 2-3 hours."
            elif intent == CustomerIntent.BILLING_INQUIRY:
                response = "I can help with your billing inquiry. Your current balance and payment options are available online."
            elif intent == CustomerIntent.COMPLAINT:
                response = "I understand your frustration. This issue has been escalated to our service team for immediate attention."
            elif intent == CustomerIntent.PAYMENT:
                # Fix the payment context retrieval
                payment_amount = self.memory.payment_history[0]['amount'] if self.memory.payment_history else 0
                response = f"Payment of ${payment_amount} is due soon. Would you like to pay now or set up autopay?"
            elif intent == CustomerIntent.SERVICE_REQUEST:
                response = "I'll help with your service request. Can you provide more details about what you need?"
            else:
                response = "I'll help with that. Please provide more details."
                
        # ADVANCED LEVEL
        elif self.current_level.name == "ADVANCED":
            print(">>> Generating ADVANCED response")
            if intent == CustomerIntent.OUTAGE_REPORT:
                area_outage = context.get('area_outage_active', False)
                if area_outage:
                    response = (f"Hello {self.memory.customer_id}, I see you're experiencing an outage. "
                            f"This is a known issue in {self.memory.location} affecting multiple customers. "
                            f"Crews are on-site. Estimated restoration: 2:30 PM.")
                else:
                    response = (f"Hello {self.memory.customer_id}, I've logged your outage report. "
                            f"I'll dispatch a crew to investigate. You'll receive updates via text.")
            elif intent == CustomerIntent.COMPLAINT:
                response = (f"I sincerely apologize for the inconvenience, {self.memory.customer_id}. "
                        f"I understand how frustrating this must be. I'm escalating this to our senior team immediately "
                        f"and applying a service credit to your account.")
            elif intent == CustomerIntent.BILLING_INQUIRY:
                response = (f"Hello {self.memory.customer_id}, I see you have a question about your bill. "
                        f"Your current balance is ${self.memory.payment_history[0]['amount']}. "
                        f"I can help explain any charges or set up a payment plan if needed.")
            elif intent == CustomerIntent.PAYMENT:
                response = (f"I can help you with your payment, {self.memory.customer_id}. "
                        f"You have ${self.memory.payment_history[0]['amount']} due in {(datetime.fromisoformat(self.memory.payment_history[0]['due_date']) - datetime.now()).days} days. "
                        f"Would you like to pay now or set up autopay for convenience?")
            else:
                response = f"Hello {self.memory.customer_id}, I'll help with your {intent.value.replace('_', ' ')}. How can I assist you today?"
                
        # PREMIUM LEVEL
        elif self.current_level.name == "PREMIUM":
            print(">>> Generating PREMIUM response")
            # Premium should use structured output, but if we're here, use fallback
            if self.has_claude_api:
                # This shouldn't happen - structured output should handle it
                response = self._generate_premium_fallback_response(message, intent, entities, context)
            else:
                # No API key - use a good fallback
                response = self._generate_premium_fallback_response(message, intent, entities, context)
            
        else:
            print(f">>> ERROR: Unknown level {self.current_level}")
            response = f"Error: Unknown processing level: {self.current_level.name}"
        
        print(f">>> Generated response: {response[:60]}...")
        
        self.processing_steps.append(ProcessingStep(
            name="Response Generation",
            description=f"Generated response using {self.current_level.value}",
            result={
                "response_length": len(response), 
                "level": self.current_level.value,
                "response_preview": response[:50] + "..."
            },
            confidence=0.7,
            processing_time=time.time() - start
        ))
        
        return response
    
    def _generate_premium_fallback_response(self, message: str, intent: CustomerIntent, entities: Dict, context: Dict) -> str:
        """Premium fallback response when API is not available - ONLY FOR PREMIUM LEVEL"""
        print(">>> Premium fallback being called")
        
        if intent in [CustomerIntent.OUTAGE_REPORT, CustomerIntent.COMPLAINT]:
            duration = "over 24 hours" if "yesterday" in message.lower() else "several hours"
            impact = " I completely understand how frustrating this must be, especially since you work from home." if entities.get("work_impact") else ""
            
            response = (f"I sincerely apologize for the extended power outage you're experiencing, {self.memory.customer_id}. "
                      f"I can see this has been affecting you for {duration}.{impact} "
                      f"\n\nGood news: I've confirmed there's an area-wide outage in {self.memory.location} affecting 2,400 customers. "
                      f"Our crews are on-site and we expect power to be restored by 2:30 PM today. "
                      f"\n\nGiven the duration and impact on your work, I'd like to offer you a $50 credit on your next bill. "
                      f"I can also set up text alerts to notify you the moment power is restored. Would you like me to do that?")
        elif intent == CustomerIntent.BILLING_INQUIRY:
            response = (f"Hello {self.memory.customer_id}, I see you have questions about your bill. "
                      f"Looking at your account, your current balance is ${self.memory.payment_history[0]['amount']} due in {(datetime.fromisoformat(self.memory.payment_history[0]['due_date']) - datetime.now()).days} days. "
                      f"I notice you've been a loyal customer with excellent payment history. "
                      f"I can provide a detailed breakdown of charges, offer budget billing to even out seasonal variations, "
                      f"or discuss our energy-saving programs that could reduce your future bills. What would be most helpful?")
        elif intent == CustomerIntent.PAYMENT:
            days_until_due = (datetime.fromisoformat(self.memory.payment_history[0]['due_date']) - datetime.now()).days
            response = (f"I'd be happy to help with your payment, {self.memory.customer_id}. "
                      f"You have ${self.memory.payment_history[0]['amount']} due in {days_until_due} days. "
                      f"As a valued customer, I can offer you several convenient options: "
                      f"pay now with instant confirmation, set up autopay to never miss a payment, "
                      f"or if needed, I can arrange a payment extension. I also see you qualify for our budget billing program "
                      f"which would make your monthly payments more predictable. Which option works best for you?")
        else:
            response = f"Hello {self.memory.customer_id}, I'm here to help. Based on your account history and current situation, how can I assist you today?"
        
        return response
    
    def _analyze_proactive_opportunities(self, message: str, intent: CustomerIntent, entities: Dict, context: Dict, sentiment: float) -> List[Dict]:
        """Proactive analysis - ALL LEVELS CAN TAKE ACTIONS"""
        start = time.time()
        actions = []
        
        print(f">>> Analyzing proactive opportunities for {self.current_level.value}")
        print(f">>> Current level name: {self.current_level.name}")
        print(f">>> Intent: {intent}, Context keys: {context.keys()}")
        
        # Debug: Check what's in context for area outage
        if 'area_outage' in context:
            print(f">>> area_outage in context: {context['area_outage']}")
        if 'area_outage_active' in context:
            print(f">>> area_outage_active in context: {context['area_outage_active']}")
        
        if self.current_level.name == "BASIC":
            # BASIC: Simple automated actions
            if intent == CustomerIntent.OUTAGE_REPORT:
                print(">>> BASIC: Creating ticket")
                # Create basic ticket
                ticket_action = Action(
                    type=ActionType.CREATE_TICKET,
                    priority="medium",
                    description="Create basic outage ticket",
                    data={"type": "outage", "location": entities.get('locations', ['Unknown'])[0] if entities.get('locations') else 'Unknown'},
                    automated=True
                )
                self._execute_action(ticket_action)
                actions.append({
                    "type": ticket_action.type.value,
                    "message": f"✅ Created ticket {ticket_action.data.get('ticket_id', 'TKT-XXX')}",
                    "priority": "medium",
                    "completed": True
                })
                
        elif self.current_level.name == "MEDIUM":
            # MEDIUM: Smart actions with ML-based decisions
            if intent == CustomerIntent.OUTAGE_REPORT:
                print(">>> MEDIUM: Creating smart ticket")
                # FIX: Check for area outage using the correct key
                area_outage_active = context.get("area_outage_active", False) or context.get("area_outage", {}).get("active", False)
                
                # Create categorized ticket with priority
                ticket_action = Action(
                    type=ActionType.CREATE_TICKET,
                    priority="high" if area_outage_active else "medium",
                    description="Create outage ticket with smart categorization",
                    data={
                        "type": "outage",
                        "location": entities.get('locations', ['Unknown'])[0] if entities.get('locations') else 'Unknown',
                        "category": "area_outage" if area_outage_active else "individual",
                        "sentiment": sentiment
                    },
                    automated=True
                )
                self._execute_action(ticket_action)
                actions.append({
                    "type": ticket_action.type.value,
                    "message": f"✅ Smart ticket created: {ticket_action.data.get('ticket_id')}",
                    "priority": ticket_action.priority,
                    "completed": True
                })
                
                # Send notification if area outage
                if area_outage_active:
                    print(">>> MEDIUM: Sending notification")
                    notify_action = Action(
                        type=ActionType.SEND_NOTIFICATION,
                        priority="high",
                        description="Send area outage notification",
                        data={"message": f"Area outage in {entities.get('locations', [self.memory.location])[0]}. Est. restoration: 2:30 PM"},
                        automated=True
                    )
                    self._execute_action(notify_action)
                    actions.append({
                        "type": notify_action.type.value,
                        "message": "✅ SMS notification sent",
                        "priority": "high",
                        "completed": True
                    })
            
            # Payment reminder for any level
            if context.get("payment_due_soon"):
                actions.append({
                    "type": "payment_reminder",
                    "message": "💳 Payment due in 2 days - setting reminder",
                    "priority": "medium",
                    "completed": False  # Requires confirmation
                })
                
        elif self.current_level.name == "ADVANCED":
            # ADVANCED: Complex multi-step actions
            if intent == CustomerIntent.OUTAGE_REPORT:
                print(">>> ADVANCED: Creating comprehensive ticket")
                # Create ticket with full context
                ticket_action = Action(
                    type=ActionType.CREATE_TICKET,
                    priority="critical" if entities.get("work_impact") else "high",
                    description="Create comprehensive outage ticket",
                    data={
                        "type": "outage",
                        "location": self.memory.location,
                        "customer_impact": "high",
                        "account": self.memory.account_number,
                        "history": len(self.memory.past_interactions)
                    },
                    automated=True
                )
                self._execute_action(ticket_action)
                actions.append({
                    "type": ticket_action.type.value,
                    "message": f"✅ Priority ticket created: {ticket_action.data.get('ticket_id')}",
                    "priority": ticket_action.priority,
                    "completed": True
                })
                
                # Multiple automated actions
                area_outage_active = context.get("area_outage_active", False)
                if area_outage_active:
                    # Auto-apply credit for extended outage
                    if "hours" in message.lower() or "yesterday" in message.lower():
                        print(">>> ADVANCED: Applying credit")
                        credit_action = Action(
                            type=ActionType.APPLY_CREDIT,
                            priority="high",
                            description="Apply outage credit",
                            data={"amount": 50, "reason": "Extended outage compensation"},
                            automated=True
                        )
                        self._execute_action(credit_action)
                        actions.append({
                            "type": credit_action.type.value,
                            "message": "✅ $50 credit applied automatically",
                            "priority": "high",
                            "completed": True
                        })
                    
                    # Schedule callback
                    print(">>> ADVANCED: Scheduling callback")
                    callback_action = Action(
                        type=ActionType.SCHEDULE_CALLBACK,
                        priority="medium",
                        description="Schedule post-restoration callback",
                        data={"time": "30 minutes after restoration"},
                        automated=True
                    )
                    self._execute_action(callback_action)
                    actions.append({
                        "type": callback_action.type.value,
                        "message": "✅ Callback scheduled for after restoration",
                        "priority": "medium",
                        "completed": True
                    })
        
        print(f">>> Total actions created: {len(actions)}")
        
        self.processing_steps.append(ProcessingStep(
            name="Action Execution",
            description=f"Executed {len([a for a in actions if a.get('completed')])} actions using {self.current_level.value}",
            result={"actions": len(actions), "types": [a['type'] for a in actions]},
            confidence=0.9,
            processing_time=time.time() - start
        ))
        
        return actions

# Streamlit UI
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .highlight-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        background-color: #f0f8ff;
        margin: 10px 0;
    }
    .proactive-action {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = MultiLevelAgent(api_key=st.session_state.api_key)
        st.session_state.messages = []
        st.session_state.comparisons = []
        st.session_state.ai_level = AILevel.BASIC
    
    # Reinitialize agent if API key changed
    if 'last_api_key' not in st.session_state:
        st.session_state.last_api_key = st.session_state.api_key
    
    if st.session_state.last_api_key != st.session_state.api_key:
        st.session_state.agent = MultiLevelAgent(api_key=st.session_state.api_key)
        st.session_state.last_api_key = st.session_state.api_key
        st.rerun()


    
    # Header - with logo
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("https://newsroom.nexteraenergy.com/images/nextera_logo.jpg", width=300)
    with col_title:
        st.title("⚡ FPL Multi-Level AI Agent Demo")
        st.markdown("### Demonstrating the 3 Pillars: 🧠 Brain (LLM) + 💾 Memory + 🎯 Goals")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # AI Level Selector
        st.markdown("### Select AI Sophistication Level")
        
        level_col1, level_col2, level_col3, level_col4 = st.columns(4)
        
        with level_col1:
            if st.button("🔤 Basic NLP", use_container_width=True, 
                        help="Keywords only - Will FAIL on 'I have not power'",
                        type="primary" if st.session_state.ai_level == AILevel.BASIC else "secondary"):
                st.session_state.ai_level = AILevel.BASIC
                st.rerun()
        
        with level_col2:
            if st.button("🧠 Advanced NLP", use_container_width=True, 
                        help="ML models - Partially understands variations",
                        type="primary" if st.session_state.ai_level == AILevel.MEDIUM else "secondary"):
                st.session_state.ai_level = AILevel.MEDIUM
                print(f">>> Button clicked: Setting ai_level to {AILevel.MEDIUM}")
                st.rerun()
        
        with level_col3:
            if st.button("🤖 Local LLM", use_container_width=True, 
                        help="Good understanding + some proactive features",
                        type="primary" if st.session_state.ai_level == AILevel.ADVANCED else "secondary"):
                st.session_state.ai_level = AILevel.ADVANCED
                st.rerun()
        
        with level_col4:
            if st.button("🌟 Claude API", use_container_width=True, 
                        help="Perfect understanding + full proactive AI",
                        type="primary" if st.session_state.ai_level == AILevel.PREMIUM else "secondary"):
                st.session_state.ai_level = AILevel.PREMIUM
                st.rerun()
        
        # Show current selection prominently
        st.markdown(f"<div class='highlight-box'>Currently Selected: <span class='big-font'>{st.session_state.ai_level.value}</span></div>", 
                   unsafe_allow_html=True)
        
        # The 3 Pillars with current state
        st.markdown("### 🏛️ The 3 AI Agent Pillars in Action")
        pillar_col1, pillar_col2, pillar_col3 = st.columns(3)
        
        with pillar_col1:
            st.markdown("#### 🧠 Brain (Changes)")
            brain_capability = {
                AILevel.BASIC.value: "❌ Can't understand 'I have not power'",
                AILevel.MEDIUM.value: "⚠️ Partially understands variations",
                AILevel.ADVANCED.value: "✅ Understands context",
                AILevel.PREMIUM.value: "💯 Perfect comprehension"
            }
            st.info(brain_capability.get(st.session_state.ai_level.value, "Unknown"))
        
        with pillar_col2:
            st.markdown("#### 💾 Memory (Persistent)")
            st.success(f"✓ Customer: {st.session_state.agent.memory.customer_id}\n"
                      f"✓ Location: {st.session_state.agent.memory.location}\n"
                      f"✓ History: {len(st.session_state.agent.memory.past_interactions)} interactions")
        
        with pillar_col3:
            st.markdown("#### 🎯 Goals (Dynamic)")
            goals_capability = {
                AILevel.BASIC.value: "📋 Basic: Answer question",
                AILevel.MEDIUM.value: "📋 Better: Confirm & respond",
                AILevel.ADVANCED.value: "📋 Smart: Multi-step resolution",
                AILevel.PREMIUM.value: "📋 Strategic: Prevent + resolve"
            }
            st.warning(goals_capability.get(st.session_state.ai_level.value, "Unknown"))
        
        # Test scenarios
        st.markdown("### 🧪 Test Scenarios - See the Difference!")
        test_col1, test_col2, test_col3 = st.columns(3)
        
        with test_col1:
            if st.button("Test: 'I have not power'", use_container_width=True,
                        help="Basic NLP will FAIL on this"):
                st.session_state.pending_message = "I have not power"
        
        with test_col2:
            if st.button("Test: 'Power out Miami Beach'", use_container_width=True,
                        help="All levels handle this"):
                st.session_state.pending_message = "Power out in miami beach for 2 hours"
        
        with test_col3:
            if st.button("Test: Complex + Emotional", use_container_width=True,
                        help="Only Premium handles perfectly"):
                st.session_state.pending_message = "My electricity has been off since yesterday and I work from home. This is unacceptable!"
        
        # Process pending message
        if 'pending_message' in st.session_state:
            message = st.session_state.pending_message
            del st.session_state.pending_message
            
            with st.spinner(f"Processing with {st.session_state.ai_level.value}..."):
                result = st.session_state.agent.process_message(message, st.session_state.ai_level)
            
            st.session_state.comparisons.append({
                'message': message,
                'ai_level': st.session_state.ai_level,
                'result': result
            })
            
            st.session_state.messages.append({"role": "user", "content": message})
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['response'],
                "metadata": result
            })
        
        # Chat display
        st.markdown("### 💬 Conversation")
        for msg in st.session_state.messages[-6:]:  # Show last 6 messages
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
                if msg["role"] == "assistant" and "metadata" in msg:
                    metadata = msg["metadata"]
                    
                    # Show key metrics
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Intent", metadata['intent'].replace('_', ' ').title())
                    with metric_cols[1]:
                        st.metric("Confidence", f"{metadata['intent_confidence']:.0%}")
                    with metric_cols[2]:
                        st.metric("Sentiment", f"{metadata['sentiment']:.2f}")
                    with metric_cols[3]:
                        # Check if this is structured output
                        if 'executed_actions' in metadata:
                            st.metric("Actions Executed", len(metadata.get('executed_actions', [])))
                        else:
                            st.metric("Actions", len(metadata['proactive_actions']))
                    
                    # Show emotional state if available (structured output)
                    if 'emotional_state' in metadata:
                        st.info(f"😊 Emotional State: {metadata['emotional_state']}")
                    
                    # Show proactive actions prominently
                    if metadata['proactive_actions']:
                        st.markdown("#### 🎯 Proactive AI Actions:")
                        for action in metadata['proactive_actions']:
                            # Check if it's from structured output
                            if 'completed' in action:
                                if action['completed']:
                                    st.success(f"✅ **{action['type'].replace('_', ' ').title()}**: {action['message']} [EXECUTED]")
                                else:
                                    st.warning(f"⏳ **{action['type'].replace('_', ' ').title()}**: {action['message']} [PENDING]")
                            else:
                                st.markdown(f"<div class='proactive-action'>{action['message']}</div>", 
                                          unsafe_allow_html=True)
                    
                    # Show structured output if available (Premium level)
                    if 'structured_output' in metadata:
                        with st.expander("🎯 Structured Output Details"):
                            tabs = st.tabs(["Actions", "Next Steps", "Internal Notes", "Full JSON"])
                            
                            with tabs[0]:
                                st.markdown("**Executed Actions:**")
                                for action in metadata.get('executed_actions', []):
                                    st.success(f"✅ {action['type']}: {action['description']}")
                                    if action.get('data'):
                                        st.json(action['data'])
                            
                            with tabs[1]:
                                st.markdown("**Recommended Next Steps:**")
                                for step in metadata.get('next_steps', []):
                                    st.write(f"• {step}")
                            
                            with tabs[2]:
                                st.markdown("**Internal Notes:**")
                                st.write(metadata.get('internal_notes', 'No notes'))
                                if metadata.get('needs_human'):
                                    st.error(f"⚠️ Needs Human: {metadata.get('escalation_reason', 'No reason provided')}")
                            
                            with tabs[3]:
                                st.json(metadata.get('structured_output', {}))
                    
                    # Regular expandable details
                    with st.expander("🔍 See Processing Details"):
                        tabs = st.tabs(["Pipeline", "Memory", "Goals"])
                        
                        with tabs[0]:
                            for step in metadata['processing_steps']:
                                st.markdown(f"**{step.name}** ({step.confidence:.0%} confidence)")
                                if isinstance(step.result, dict):
                                    st.json(step.result)
                                else:
                                    st.write(step.result)
                        
                        with tabs[1]:
                            st.json(metadata.get('memory_context', {}))
                        
                        with tabs[2]:
                            for goal in metadata.get('active_goals', []):
                                st.write(f"• {goal['goal']} - {goal['status']}")
        
        # Input
        user_input = st.chat_input("Type your message...")
        if user_input:
            st.session_state.pending_message = user_input
            st.rerun()
    
    with col2:
        # Current AI Level Pipeline
        st.markdown(f"### 🔧 {st.session_state.ai_level.value}")
        st.markdown("**Processing Pipeline:**")
        
        pipeline_steps = st.session_state.agent.get_pipeline_steps(st.session_state.ai_level)
        for i, step in enumerate(pipeline_steps):
            st.markdown(f"{i+1}. {step['icon']} **{step['step']}**")
            st.caption(step['description'])
            if i < len(pipeline_steps) - 1:
                st.markdown("↓")
        
        st.markdown("---")
        
        # Capability Chart - FIXED to show actual differences
        st.markdown("### 📊 AI Capabilities")
        
        capabilities_data = {
            'Capability': ['Intent Accuracy', 'Context Understanding', 'Proactive Features', 'Response Quality'],
            st.session_state.ai_level.value: {
                AILevel.BASIC.value: [60, 40, 20, 50],
                AILevel.MEDIUM.value: [75, 65, 50, 70],
                AILevel.ADVANCED.value: [85, 80, 75, 85],
                AILevel.PREMIUM.value: [98, 95, 90, 98]
            }.get(st.session_state.ai_level.value, [50, 50, 50, 50])
        }
        
        # Create radar chart for current level only
        fig_radar = go.Figure()
        
        color_map = {
            AILevel.BASIC.value: 'red',
            AILevel.MEDIUM.value: 'orange',
            AILevel.ADVANCED.value: 'blue',
            AILevel.PREMIUM.value: 'green'
        }
        
        fig_radar.add_trace(go.Scatterpolar(
            r=capabilities_data[st.session_state.ai_level.value],
            theta=capabilities_data['Capability'],
            fill='toself',
            name=st.session_state.ai_level.value,
            line=dict(color=color_map.get(st.session_state.ai_level.value, 'gray'))
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            height=300,
            title=f"{st.session_state.ai_level.value} Capabilities"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Memory State
        st.markdown("### 💾 Current Memory State")
        memory_metrics = {
            "Interactions": len(st.session_state.agent.memory.conversation_history),
            "Sentiment Trend": "😟 Declining" if len(st.session_state.agent.memory.sentiment_trend) > 2 and 
                              sum(st.session_state.agent.memory.sentiment_trend[-3:]) / 3 < -0.3 else "😊 Stable",
            "Area Outage": "🔴 Active" if st.session_state.agent.memory.area_outages.get("miami beach", {}).get("active") else "🟢 Clear",
            "Actions Executed": len(st.session_state.agent.memory.executed_actions)
        }
        
        for key, value in memory_metrics.items():
            st.metric(key, value)
        
        # Show executed actions log
        if st.session_state.agent.memory.executed_actions:
            with st.expander("⚡ Executed Actions Log"):
                for action in st.session_state.agent.memory.executed_actions[-5:]:  # Show last 5
                    st.success(f"✅ {action.type.value}: {action.description}")
                    st.caption(f"Executed at: {action.timestamp}")
                    if action.data:
                        st.json(action.data)
    
    # Comparison section
    if len(st.session_state.comparisons) >= 2:
        st.markdown("---")
        st.markdown("### 🔄 Side-by-Side Comparison - See the Differences!")
        
        # Get unique AI levels from comparisons
        comparison_df = pd.DataFrame([
            {
                'AI Level': comp['ai_level'].value,
                'Message': comp['message'][:30] + "...",
                'Intent': comp['result']['intent'],
                'Confidence': comp['result']['intent_confidence'],
                'Proactive Actions': len(comp['result']['proactive_actions']),
                'Response Quality': {
                    AILevel.BASIC.value: "Generic",
                    AILevel.MEDIUM.value: "Better",
                    AILevel.ADVANCED.value: "Good",
                    AILevel.PREMIUM.value: "Excellent"
                }.get(comp['ai_level'].value, "Unknown")
            }
            for comp in st.session_state.comparisons[-4:]
        ])
        
        st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main()
