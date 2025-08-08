import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from app import db
from models import AILearningData, Sale, Conversation, Customer
import json
import math

logger = logging.getLogger(__name__)

class ReinforcementLearner:
    def __init__(self):
        # Learning parameters
        self.exploration_rate = 0.2  # 20% chance to try less successful strategies
        self.learning_decay = 0.95   # Decay rate for exploration over time
        
        # Available strategies
        self.strategies = ["consultivo", "escassez", "emocional", "racional"]
        
        # Initialize strategies if not exists
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize learning data for all strategies if they don't exist"""
        try:
            for strategy in self.strategies:
                existing = AILearningData.query.filter_by(strategy_name=strategy).first()
                if not existing:
                    learning_data = AILearningData(
                        strategy_name=strategy,
                        success_count=0,
                        total_attempts=1,  # Start with 1 to avoid division by zero
                        success_rate=0.25,  # Equal probability initially
                        context_keywords="{}",
                        customer_sentiment=0.0,
                        message_sequence="[]"
                    )
                    db.session.add(learning_data)
            
            db.session.commit()
            logger.info("Initialized AI learning strategies")
            
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
            db.session.rollback()
    
    def get_best_strategy(self, customer_analysis: Dict[str, Any]) -> str:
        """Get the best strategy based on current learning and customer context"""
        try:
            # Get all learning data
            learning_data = AILearningData.query.all()
            
            if not learning_data:
                self._initialize_strategies()
                return "consultivo"  # Default strategy
            
            # Calculate exploration vs exploitation
            total_attempts = sum([data.total_attempts for data in learning_data])
            current_exploration_rate = self.exploration_rate * (self.learning_decay ** (total_attempts / 100))
            
            # Decide whether to explore or exploit
            import random
            if random.random() < current_exploration_rate:
                # Exploration: choose strategy with lower success rate to learn more
                strategy = self._choose_exploration_strategy(learning_data, customer_analysis)
                logger.info(f"Exploration mode: chose strategy {strategy}")
            else:
                # Exploitation: choose best performing strategy
                strategy = self._choose_best_strategy(learning_data, customer_analysis)
                logger.info(f"Exploitation mode: chose strategy {strategy}")
            
            # Update attempt count
            self._update_attempt_count(strategy)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error getting best strategy: {e}")
            return "consultivo"
    
    def _choose_best_strategy(self, learning_data: List[AILearningData], 
                            customer_analysis: Dict[str, Any]) -> str:
        """Choose the strategy with highest success rate for similar contexts"""
        try:
            # Score each strategy based on success rate and context similarity
            strategy_scores = {}
            
            for data in learning_data:
                base_score = data.success_rate
                
                # Bonus for context similarity
                context_bonus = self._calculate_context_similarity(data, customer_analysis)
                
                # Confidence bonus (more attempts = more confidence)
                confidence_bonus = min(0.1, data.total_attempts / 100)
                
                final_score = base_score + context_bonus + confidence_bonus
                strategy_scores[data.strategy_name] = final_score
            
            # Return strategy with highest score
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            return best_strategy
            
        except Exception as e:
            logger.error(f"Error choosing best strategy: {e}")
            return "consultivo"
    
    def _choose_exploration_strategy(self, learning_data: List[AILearningData], 
                                   customer_analysis: Dict[str, Any]) -> str:
        """Choose a strategy to explore (lower success rate strategies prioritized)"""
        try:
            # Weight strategies inversely by their success rate (less successful = higher weight)
            weights = []
            strategies = []
            
            for data in learning_data:
                # Inverse weight: lower success rate = higher exploration weight
                weight = 1.0 / (data.success_rate + 0.1)  # Add 0.1 to avoid division by zero
                weights.append(weight)
                strategies.append(data.strategy_name)
            
            # Weighted random selection
            import random
            return random.choices(strategies, weights=weights, k=1)[0]
            
        except Exception as e:
            logger.error(f"Error choosing exploration strategy: {e}")
            return "consultivo"
    
    def _calculate_context_similarity(self, learning_data: AILearningData, 
                                    customer_analysis: Dict[str, Any]) -> float:
        """Calculate similarity bonus based on customer context"""
        try:
            bonus = 0.0
            
            # Parse stored context
            stored_keywords = json.loads(learning_data.context_keywords or "{}")
            current_keywords = customer_analysis.get('keywords', [])
            
            # Keyword similarity
            if stored_keywords and current_keywords:
                common_keywords = set(stored_keywords.keys()).intersection(set(current_keywords))
                bonus += len(common_keywords) * 0.02  # 2% bonus per common keyword
            
            # Sentiment similarity
            if learning_data.customer_sentiment:
                sentiment_diff = abs(learning_data.customer_sentiment - customer_analysis.get('sentiment', 0))
                sentiment_bonus = max(0, (1 - sentiment_diff) * 0.05)  # Up to 5% bonus
                bonus += sentiment_bonus
            
            return min(bonus, 0.15)  # Cap at 15% bonus
            
        except Exception as e:
            logger.error(f"Error calculating context similarity: {e}")
            return 0.0
    
    def _update_attempt_count(self, strategy: str):
        """Update attempt count for a strategy"""
        try:
            learning_data = AILearningData.query.filter_by(strategy_name=strategy).first()
            if learning_data:
                learning_data.total_attempts += 1
                learning_data.last_updated = datetime.utcnow()
                db.session.commit()
        except Exception as e:
            logger.error(f"Error updating attempt count: {e}")
            db.session.rollback()
    
    def record_success(self, customer_id: int, strategy: str, conversation_messages: int):
        """Record a successful sale and update learning data"""
        try:
            # Get customer and recent conversation data
            customer = Customer.query.get(customer_id)
            if not customer:
                logger.error(f"Customer {customer_id} not found for success recording")
                return
            
            # Get recent conversation context
            recent_conversations = Conversation.query.filter_by(
                customer_id=customer_id
            ).order_by(Conversation.timestamp.desc()).limit(10).all()
            
            # Extract context data
            context_keywords = {}
            sentiments = []
            message_sequence = []
            
            for conv in reversed(recent_conversations):  # Reverse to get chronological order
                if conv.sentiment_score:
                    sentiments.append(conv.sentiment_score)
                message_sequence.append(conv.message_type)
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            
            # Update learning data
            learning_data = AILearningData.query.filter_by(strategy_name=strategy).first()
            if learning_data:
                learning_data.success_count += 1
                learning_data.success_rate = learning_data.success_count / learning_data.total_attempts
                learning_data.customer_sentiment = avg_sentiment
                learning_data.context_keywords = json.dumps(context_keywords)
                learning_data.message_sequence = json.dumps(message_sequence)
                learning_data.last_updated = datetime.utcnow()
            
            db.session.commit()
            logger.info(f"Recorded success for strategy {strategy}. New success rate: {learning_data.success_rate:.3f}")
            
        except Exception as e:
            logger.error(f"Error recording success: {e}")
            db.session.rollback()
    
    def record_failure(self, customer_id: int, strategy: str, reason: str = "no_purchase"):
        """Record a failed attempt (no sale) and update learning data"""
        try:
            # Note: Attempt count is already incremented when strategy is chosen
            # This method is for additional failure analysis if needed
            
            learning_data = AILearningData.query.filter_by(strategy_name=strategy).first()
            if learning_data:
                # Recalculate success rate (success_count stays the same, total_attempts already incremented)
                learning_data.success_rate = learning_data.success_count / learning_data.total_attempts
                learning_data.last_updated = datetime.utcnow()
            
            db.session.commit()
            logger.info(f"Updated failure data for strategy {strategy}. Success rate: {learning_data.success_rate:.3f}")
            
        except Exception as e:
            logger.error(f"Error recording failure: {e}")
            db.session.rollback()
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get current learning statistics for all strategies"""
        try:
            learning_data = AILearningData.query.all()
            
            stats = {
                "strategies": {},
                "total_attempts": 0,
                "total_successes": 0,
                "overall_success_rate": 0.0,
                "best_strategy": "consultivo",
                "worst_strategy": "consultivo"
            }
            
            best_rate = 0.0
            worst_rate = 1.0
            
            for data in learning_data:
                strategy_stats = {
                    "success_count": data.success_count,
                    "total_attempts": data.total_attempts,
                    "success_rate": data.success_rate,
                    "last_updated": data.last_updated.isoformat() if data.last_updated else None
                }
                
                stats["strategies"][data.strategy_name] = strategy_stats
                stats["total_attempts"] += data.total_attempts
                stats["total_successes"] += data.success_count
                
                if data.success_rate > best_rate:
                    best_rate = data.success_rate
                    stats["best_strategy"] = data.strategy_name
                
                if data.success_rate < worst_rate:
                    worst_rate = data.success_rate
                    stats["worst_strategy"] = data.strategy_name
            
            if stats["total_attempts"] > 0:
                stats["overall_success_rate"] = stats["total_successes"] / stats["total_attempts"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {"error": str(e)}
