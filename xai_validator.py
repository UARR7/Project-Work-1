import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import shap
import lime
from lime.lime_text import LimeTextExplainer
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import warnings

from docs_generator import DocsGenerator
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class XAIValidationResult:
    """Enhanced validation result with XAI explanations"""
    element_name: str
    file_path: str
    validation_type: str
    score: float
    explanation: Dict[str, Any]
    feature_importance: Dict[str, float]
    lime_explanation: Optional[Dict] = None
    shap_values: Optional[np.ndarray] = None
    confidence: float = 0.0
    recommendations: List[str] = None

class XAIDocumentationValidator:
    """XAI-enhanced documentation validator using SHAP and LIME for explainable validation"""
    
    def __init__(self, use_transformers=True):
        self.use_transformers = use_transformers
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.quality_classifier = None
        self.lime_explainer = LimeTextExplainer(class_names=['Poor', 'Good'])
        
        # Initialize transformer-based models if available
        if use_transformers:
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                 model="cardiffnlp/twitter-roberta-base-sentiment-latest")
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")
            except Exception as e:
                logger.warning(f"Failed to load transformers: {e}")
                self.use_transformers = False
        
        self.validation_results = []
        self._prepare_quality_model()
    
    def _prepare_quality_model(self):
        """Prepare a machine learning model to assess documentation quality"""
        # Synthetic training data for documentation quality assessment
        training_data = self._generate_training_data()
        
        X_train = self.vectorizer.fit_transform(training_data['descriptions'])
        y_train = training_data['quality_labels']
        
        self.quality_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.quality_classifier.fit(X_train, y_train)
        
        # Initialize SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.quality_classifier)
    
    def _generate_training_data(self) -> Dict[str, Any]:
        """Generate synthetic training data for quality assessment"""
        good_descriptions = [
            "Calculates the factorial of a non-negative integer using recursive approach",
            "Validates user input email address format using regex pattern matching",
            "Retrieves user authentication token from secure session storage",
            "Processes payment transaction with comprehensive error handling and logging",
            "Implements binary search algorithm for efficient data retrieval",
            "Manages database connection pooling with automatic retry mechanism",
            "Transforms raw JSON data into structured pandas DataFrame for analysis",
            "Generates secure random password with configurable complexity requirements",
            "Handles file upload with virus scanning and size validation",
            "Optimizes database queries using prepared statements and caching"
        ]
        
        poor_descriptions = [
            "Function that handles stuff",
            "Does something with data",
            "A method",
            "Processes things",
            "Function that does things",
            "Handles user stuff",
            "Method for processing",
            "A function",
            "Does operations",
            "Handles some functionality"
        ]
        
        descriptions = good_descriptions + poor_descriptions
        quality_labels = [1] * len(good_descriptions) + [0] * len(poor_descriptions)
        
        return {
            'descriptions': descriptions,
            'quality_labels': quality_labels
        }
#main 
    def validate_with_xai(self, repo_info: Dict[str, Any]) -> List[XAIValidationResult]:
        """Main XAI validation function"""
        results = []
        
        for file_info in repo_info['files']:
            if file_info['extension'] in ['.py', '.js', '.ts', '.java']:
                # Validate functions with XAI
                for func in file_info.get('functions', []):
                    result = self._validate_function_with_xai(func, file_info)
                    if result:
                        results.append(result)
                
                # Validate classes with XAI
                for cls in file_info.get('classes', []):
                    result = self._validate_class_with_xai(cls, file_info)
                    if result:
                        results.append(result)
                
                # Validate file summary
                if file_info.get('ai_summary'):
                    result = self._validate_summary_with_xai(file_info)
                    if result:
                        results.append(result)
        
        return results
    
    # def _validate_function_with_xai(self, func: Dict, file_info: Dict) -> Optional[XAIValidationResult]:
    #     """Validate function documentation using XAI techniques"""
    #     ai_description = func.get('ai_description', '')
    #     if not ai_description:
    #         return None
        
    #     # Feature extraction
    #     features = self._extract_function_features(func, file_info)
        
    #     # Quality prediction with SHAP explanation
    #     X_test = self.vectorizer.transform([ai_description])
    #     quality_prob = self.quality_classifier.predict_proba(X_test)[0]
    #     quality_score = quality_prob[1]  # Probability of being good quality
        
    #     # SHAP explanation
    #     shap_values = self.shap_explainer.shap_values(X_test.toarray())[1]  # For class 1 (good quality)
        
    #     # LIME explanation
    #     lime_explanation = self._get_lime_explanation(ai_description)
        
    #     # Feature importance from model
    #     feature_names = self.vectorizer.get_feature_names_out()
    #     feature_importance = {}
        
    #     # Get top contributing features
    #     feature_indices = X_test.toarray()[0].nonzero()[0]
    #     for idx in feature_indices:
    #         if abs(shap_values[0][idx]) > 0.001:  # Only significant features
    #             feature_importance[feature_names[idx]] = float(shap_values[0][idx])
        
    #     # Generate recommendations
    #     recommendations = self._generate_function_recommendations(
    #         func, quality_score, feature_importance, lime_explanation
    #     )
        
    #     return XAIValidationResult(
    #         element_name=func['name'],
    #         file_path=file_info['rel_path'],
    #         validation_type='function_description',
    #         score=quality_score,
    #         explanation={
    #             'quality_probability': quality_prob.tolist(),
    #             'top_features': dict(sorted(feature_importance.items(), 
    #                                       key=lambda x: abs(x[1]), reverse=True)[:10])
    #         },
    #         feature_importance=feature_importance,
    #         lime_explanation=lime_explanation,
    #         shap_values=shap_values,
    #         confidence=self._calculate_confidence(quality_score, shap_values),
    #         recommendations=recommendations
    #     )
    
    # def _validate_class_with_xai(self, cls: Dict, file_info: Dict) -> Optional[XAIValidationResult]:
    #     """Validate class documentation using XAI techniques"""
    #     ai_description = cls.get('ai_description', '')
    #     if not ai_description:
    #         return None
        
    #     # Similar to function validation but with class-specific features
    #     features = self._extract_class_features(cls, file_info)
        
    #     X_test = self.vectorizer.transform([ai_description])
    #     quality_prob = self.quality_classifier.predict_proba(X_test)[0]
    #     quality_score = quality_prob[1]
        
    #     shap_values = self.shap_explainer.shap_values(X_test.toarray())[1]
    #     lime_explanation = self._get_lime_explanation(ai_description)
        
    #     feature_names = self.vectorizer.get_feature_names_out()
    #     feature_importance = {}
    #     feature_indices = X_test.toarray()[0].nonzero()[0]
        
    #     for idx in feature_indices:
    #         if abs(shap_values[0][idx]) > 0.001:
    #             feature_importance[feature_names[idx]] = float(shap_values[0][idx])
        
    #     recommendations = self._generate_class_recommendations(
    #         cls, quality_score, feature_importance, lime_explanation
    #     )
        
    #     return XAIValidationResult(
    #         element_name=cls['name'],
    #         file_path=file_info['rel_path'],
    #         validation_type='class_description',
    #         score=quality_score,
    #         explanation={
    #             'quality_probability': quality_prob.tolist(),
    #             'top_features': dict(sorted(feature_importance.items(), 
    #                                       key=lambda x: abs(x[1]), reverse=True)[:10])
    #         },
    #         feature_importance=feature_importance,
    #         lime_explanation=lime_explanation,
    #         shap_values=shap_values,
    #         confidence=self._calculate_confidence(quality_score, shap_values),
    #         recommendations=recommendations
    #     )
    
    # def _validate_summary_with_xai(self, file_info: Dict) -> Optional[XAIValidationResult]:
    #     """Validate file summary using XAI techniques"""
    #     ai_summary = file_info.get('ai_summary', '')
    #     if not ai_summary:
    #         return None
        
    #     # Enhanced validation for file summaries
    #     features = self._extract_summary_features(file_info)
        
    #     X_test = self.vectorizer.transform([ai_summary])
    #     quality_prob = self.quality_classifier.predict_proba(X_test)[0]
    #     quality_score = quality_prob[1]
        
    #     shap_values = self.shap_explainer.shap_values(X_test.toarray())[1]
    #     lime_explanation = self._get_lime_explanation(ai_summary)
        
    #     # Semantic coherence check
    #     coherence_score = self._check_semantic_coherence(ai_summary, file_info)
    #     combined_score = (quality_score + coherence_score) / 2
        
    #     feature_names = self.vectorizer.get_feature_names_out()
    #     feature_importance = {}
    #     feature_indices = X_test.toarray()[0].nonzero()[0]
        
    #     for idx in feature_indices:
    #         if abs(shap_values[0][idx]) > 0.001:
    #             feature_importance[feature_names[idx]] = float(shap_values[0][idx])
        
    #     recommendations = self._generate_summary_recommendations(
    #         file_info, combined_score, feature_importance, lime_explanation
    #     )
        
    #     return XAIValidationResult(
    #         element_name=f"File Summary ({file_info['name']})",
    #         file_path=file_info['rel_path'],
    #         validation_type='file_summary',
    #         score=combined_score,
    #         explanation={
    #             'quality_probability': quality_prob.tolist(),
    #             'coherence_score': coherence_score,
    #             'top_features': dict(sorted(feature_importance.items(), 
    #                                       key=lambda x: abs(x[1]), reverse=True)[:10])
    #         },
    #         feature_importance=feature_importance,
    #         lime_explanation=lime_explanation,
    #         shap_values=shap_values,
    #         confidence=self._calculate_confidence(combined_score, shap_values),
    #         recommendations=recommendations
    #     )

    # def _validate_function_with_xai(self, func: Dict, file_info: Dict) -> Optional[XAIValidationResult]:
    #     """Validate function documentation using XAI techniques"""
    #     ai_description = func.get('ai_description', '')
    #     if not ai_description:
    #         return None
        
    #     # Feature extraction
    #     features = self._extract_function_features(func, file_info)
        
    #     # Quality prediction with SHAP explanation
    #     X_test = self.vectorizer.transform([ai_description])
    #     quality_prob = self.quality_classifier.predict_proba(X_test)[0]
    #     quality_score = quality_prob[1] if len(quality_prob) > 1 else quality_prob[0]  # Handle single class case
        
    #     # SHAP explanation - handle both binary and single class cases
    #     shap_values = self.shap_explainer.shap_values(X_test.toarray())
    #     if isinstance(shap_values, list) and len(shap_values) > 1:
    #         # Binary classification case
    #         shap_values_for_positive = shap_values[1]
    #     else:
    #         # Single class or regression case
    #         shap_values_for_positive = shap_values if not isinstance(shap_values, list) else shap_values[0]
        
    #     # LIME explanation
    #     lime_explanation = self._get_lime_explanation(ai_description)
        
    #     # Feature importance from model
    #     feature_names = self.vectorizer.get_feature_names_out()
    #     feature_importance = {}
        
    #     # Get top contributing features
    #     X_test_array = X_test.toarray()[0]
    #     feature_indices = X_test_array.nonzero()[0]
        
    #     for idx in feature_indices:
    #         if idx < len(shap_values_for_positive[0]) and abs(shap_values_for_positive[0][idx]) > 0.001:
    #             feature_importance[feature_names[idx]] = float(shap_values_for_positive[0][idx])
        
    #     # Generate recommendations
    #     recommendations = self._generate_function_recommendations(
    #         func, quality_score, feature_importance, lime_explanation
    #     )
        
    #     return XAIValidationResult(
    #         element_name=func['name'],
    #         file_path=file_info['rel_path'],
    #         validation_type='function_description',
    #         score=quality_score,
    #         explanation={
    #             'quality_probability': quality_prob.tolist(),
    #             'top_features': dict(sorted(feature_importance.items(), 
    #                                       key=lambda x: abs(x[1]), reverse=True)[:10])
    #         },
    #         feature_importance=feature_importance,
    #         lime_explanation=lime_explanation,
    #         shap_values=shap_values_for_positive,
    #         confidence=self._calculate_confidence(quality_score, shap_values_for_positive),
    #         recommendations=recommendations
    #     )
    
    # def _validate_class_with_xai(self, cls: Dict, file_info: Dict) -> Optional[XAIValidationResult]:
    #     """Validate class documentation using XAI techniques"""
    #     ai_description = cls.get('ai_description', '')
    #     if not ai_description:
    #         return None
        
    #     # Similar to function validation but with class-specific features
    #     features = self._extract_class_features(cls, file_info)
        
    #     X_test = self.vectorizer.transform([ai_description])
    #     quality_prob = self.quality_classifier.predict_proba(X_test)[0]
    #     quality_score = quality_prob[1] if len(quality_prob) > 1 else quality_prob[0]
        
    #     # Handle SHAP values safely
    #     shap_values = self.shap_explainer.shap_values(X_test.toarray())
    #     if isinstance(shap_values, list) and len(shap_values) > 1:
    #         shap_values_for_positive = shap_values[1]
    #     else:
    #         shap_values_for_positive = shap_values if not isinstance(shap_values, list) else shap_values[0]
        
    #     lime_explanation = self._get_lime_explanation(ai_description)
        
    #     feature_names = self.vectorizer.get_feature_names_out()
    #     feature_importance = {}
    #     X_test_array = X_test.toarray()[0]
    #     feature_indices = X_test_array.nonzero()[0]
        
    #     for idx in feature_indices:
    #         if idx < len(shap_values_for_positive[0]) and abs(shap_values_for_positive[0][idx]) > 0.001:
    #             feature_importance[feature_names[idx]] = float(shap_values_for_positive[0][idx])
        
    #     recommendations = self._generate_class_recommendations(
    #         cls, quality_score, feature_importance, lime_explanation
    #     )
        
    #     return XAIValidationResult(
    #         element_name=cls['name'],
    #         file_path=file_info['rel_path'],
    #         validation_type='class_description',
    #         score=quality_score,
    #         explanation={
    #             'quality_probability': quality_prob.tolist(),
    #             'top_features': dict(sorted(feature_importance.items(), 
    #                                       key=lambda x: abs(x[1]), reverse=True)[:10])
    #         },
    #         feature_importance=feature_importance,
    #         lime_explanation=lime_explanation,
    #         shap_values=shap_values_for_positive,
    #         confidence=self._calculate_confidence(quality_score, shap_values_for_positive),
    #         recommendations=recommendations
    #     )
    
    # def _validate_summary_with_xai(self, file_info: Dict) -> Optional[XAIValidationResult]:
    #     """Validate file summary using XAI techniques"""
    #     ai_summary = file_info.get('ai_summary', '')
    #     if not ai_summary:
    #         return None
        
    #     # Enhanced validation for file summaries
    #     features = self._extract_summary_features(file_info)
        
    #     X_test = self.vectorizer.transform([ai_summary])
    #     quality_prob = self.quality_classifier.predict_proba(X_test)[0]
    #     quality_score = quality_prob[1] if len(quality_prob) > 1 else quality_prob[0]
        
    #     # Handle SHAP values safely
    #     shap_values = self.shap_explainer.shap_values(X_test.toarray())
    #     if isinstance(shap_values, list) and len(shap_values) > 1:
    #         shap_values_for_positive = shap_values[1]
    #     else:
    #         shap_values_for_positive = shap_values if not isinstance(shap_values, list) else shap_values[0]
        
    #     lime_explanation = self._get_lime_explanation(ai_summary)
        
    #     # Semantic coherence check
    #     coherence_score = self._check_semantic_coherence(ai_summary, file_info)
    #     combined_score = (quality_score + coherence_score) / 2
        
    #     feature_names = self.vectorizer.get_feature_names_out()
    #     feature_importance = {}
    #     X_test_array = X_test.toarray()[0]
    #     feature_indices = X_test_array.nonzero()[0]
        
    #     for idx in feature_indices:
    #         if idx < len(shap_values_for_positive[0]) and abs(shap_values_for_positive[0][idx]) > 0.001:
    #             feature_importance[feature_names[idx]] = float(shap_values_for_positive[0][idx])
        
    #     recommendations = self._generate_summary_recommendations(
    #         file_info, combined_score, feature_importance, lime_explanation
    #     )
        
    #     return XAIValidationResult(
    #         element_name=f"File Summary ({file_info['name']})",
    #         file_path=file_info['rel_path'],
    #         validation_type='file_summary',
    #         score=combined_score,
    #         explanation={
    #             'quality_probability': quality_prob.tolist(),
    #             'coherence_score': coherence_score,
    #             'top_features': dict(sorted(feature_importance.items(), 
    #                                       key=lambda x: abs(x[1]), reverse=True)[:10])
    #         },
    #         feature_importance=feature_importance,
    #         lime_explanation=lime_explanation,
    #         shap_values=shap_values_for_positive,
    #         confidence=self._calculate_confidence(combined_score, shap_values_for_positive),
    #         recommendations=recommendations
    #     )

    def _validate_function_with_xai(self, func: Dict, file_info: Dict) -> Optional[XAIValidationResult]:
        """Validate function documentation using XAI techniques"""
        ai_description = func.get('ai_description', '')
        if not ai_description:
            return None
        
        # Feature extraction
        features = self._extract_function_features(func, file_info)
        
        # Quality prediction with SHAP explanation
        X_test = self.vectorizer.transform([ai_description])
        quality_prob = self.quality_classifier.predict_proba(X_test)[0]
        quality_score = quality_prob[1] if len(quality_prob) > 1 else quality_prob[0]  # Handle single class case
        
        # SHAP explanation - handle both binary and single class cases
        shap_values = self.shap_explainer.shap_values(X_test.toarray())
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # Binary classification case
            shap_values_for_positive = shap_values[1]
        else:
            # Single class or regression case
            shap_values_for_positive = shap_values if not isinstance(shap_values, list) else shap_values[0]
        
        # LIME explanation
        lime_explanation = self._get_lime_explanation(ai_description)
        
        # Feature importance from model
        feature_names = self.vectorizer.get_feature_names_out()
        feature_importance = {}
        
        # Get top contributing features
        X_test_array = X_test.toarray()[0]
        feature_indices = X_test_array.nonzero()[0]
        
        # Safely handle SHAP values array
        try:
            if len(shap_values_for_positive.shape) > 1:
                shap_vals = shap_values_for_positive[0]
            else:
                shap_vals = shap_values_for_positive
                
            for idx in feature_indices:
                if idx < len(shap_vals) and abs(float(shap_vals[idx])) > 0.001:
                    feature_importance[feature_names[idx]] = float(shap_vals[idx])
        except (IndexError, AttributeError, TypeError):
            # Fallback if SHAP values are problematic
            logger.warning("Could not extract SHAP feature importance, using basic importance")
            for idx in feature_indices[:10]:  # Just use top features
                feature_importance[feature_names[idx]] = float(X_test_array[idx])
        
        # Generate recommendations
        recommendations = self._generate_function_recommendations(
            func, quality_score, feature_importance, lime_explanation
        )
        
        return XAIValidationResult(
            element_name=func['name'],
            file_path=file_info['rel_path'],
            validation_type='function_description',
            score=quality_score,
            explanation={
                'quality_probability': quality_prob.tolist(),
                'top_features': dict(sorted(feature_importance.items(), 
                                          key=lambda x: abs(x[1]), reverse=True)[:10])
            },
            feature_importance=feature_importance,
            lime_explanation=lime_explanation,
            shap_values=shap_values_for_positive,
            confidence=self._calculate_confidence(quality_score, shap_values_for_positive),
            recommendations=recommendations
        )
    
    def _validate_class_with_xai(self, cls: Dict, file_info: Dict) -> Optional[XAIValidationResult]:
        """Validate class documentation using XAI techniques"""
        ai_description = cls.get('ai_description', '')
        if not ai_description:
            return None
        
        # Similar to function validation but with class-specific features
        features = self._extract_class_features(cls, file_info)
        
        X_test = self.vectorizer.transform([ai_description])
        quality_prob = self.quality_classifier.predict_proba(X_test)[0]
        quality_score = quality_prob[1] if len(quality_prob) > 1 else quality_prob[0]
        
        # Handle SHAP values safely
        shap_values = self.shap_explainer.shap_values(X_test.toarray())
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_for_positive = shap_values[1]
        else:
            shap_values_for_positive = shap_values if not isinstance(shap_values, list) else shap_values[0]
        
        lime_explanation = self._get_lime_explanation(ai_description)
        
        feature_names = self.vectorizer.get_feature_names_out()
        feature_importance = {}
        X_test_array = X_test.toarray()[0]
        feature_indices = X_test_array.nonzero()[0]
        
        # Safely handle SHAP values array
        try:
            if len(shap_values_for_positive.shape) > 1:
                shap_vals = shap_values_for_positive[0]
            else:
                shap_vals = shap_values_for_positive
                
            for idx in feature_indices:
                if idx < len(shap_vals) and abs(float(shap_vals[idx])) > 0.001:
                    feature_importance[feature_names[idx]] = float(shap_vals[idx])
        except (IndexError, AttributeError, TypeError):
            # Fallback if SHAP values are problematic
            for idx in feature_indices[:10]:  # Just use top features
                feature_importance[feature_names[idx]] = float(X_test_array[idx])
        
        recommendations = self._generate_class_recommendations(
            cls, quality_score, feature_importance, lime_explanation
        )
        
        return XAIValidationResult(
            element_name=cls['name'],
            file_path=file_info['rel_path'],
            validation_type='class_description',
            score=quality_score,
            explanation={
                'quality_probability': quality_prob.tolist(),
                'top_features': dict(sorted(feature_importance.items(), 
                                          key=lambda x: abs(x[1]), reverse=True)[:10])
            },
            feature_importance=feature_importance,
            lime_explanation=lime_explanation,
            shap_values=shap_values_for_positive,
            confidence=self._calculate_confidence(quality_score, shap_values_for_positive),
            recommendations=recommendations
        )
    
    def _validate_summary_with_xai(self, file_info: Dict) -> Optional[XAIValidationResult]:
        """Validate file summary using XAI techniques"""
        ai_summary = file_info.get('ai_summary', '')
        if not ai_summary:
            return None
        
        # Enhanced validation for file summaries
        features = self._extract_summary_features(file_info)
        
        X_test = self.vectorizer.transform([ai_summary])
        quality_prob = self.quality_classifier.predict_proba(X_test)[0]
        quality_score = quality_prob[1] if len(quality_prob) > 1 else quality_prob[0]
        
        # Handle SHAP values safely
        shap_values = self.shap_explainer.shap_values(X_test.toarray())
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_for_positive = shap_values[1]
        else:
            shap_values_for_positive = shap_values if not isinstance(shap_values, list) else shap_values[0]
        
        lime_explanation = self._get_lime_explanation(ai_summary)
        
        # Semantic coherence check
        coherence_score = self._check_semantic_coherence(ai_summary, file_info)
        combined_score = (quality_score + coherence_score) / 2
        
        feature_names = self.vectorizer.get_feature_names_out()
        feature_importance = {}
        X_test_array = X_test.toarray()[0]
        feature_indices = X_test_array.nonzero()[0]
        
        # Safely handle SHAP values array
        try:
            if len(shap_values_for_positive.shape) > 1:
                shap_vals = shap_values_for_positive[0]
            else:
                shap_vals = shap_values_for_positive
                
            for idx in feature_indices:
                if idx < len(shap_vals) and abs(float(shap_vals[idx])) > 0.001:
                    feature_importance[feature_names[idx]] = float(shap_vals[idx])
        except (IndexError, AttributeError, TypeError):
            # Fallback if SHAP values are problematic
            for idx in feature_indices[:10]:  # Just use top features
                feature_importance[feature_names[idx]] = float(X_test_array[idx])
        
        recommendations = self._generate_summary_recommendations(
            file_info, combined_score, feature_importance, lime_explanation
        )
        
        return XAIValidationResult(
            element_name=f"File Summary ({file_info['name']})",
            file_path=file_info['rel_path'],
            validation_type='file_summary',
            score=combined_score,
            explanation={
                'quality_probability': quality_prob.tolist(),
                'coherence_score': coherence_score,
                'top_features': dict(sorted(feature_importance.items(), 
                                          key=lambda x: abs(x[1]), reverse=True)[:10])
            },
            feature_importance=feature_importance,
            lime_explanation=lime_explanation,
            shap_values=shap_values_for_positive,
            confidence=self._calculate_confidence(combined_score, shap_values_for_positive),
            recommendations=recommendations
        )
    
    def _get_lime_explanation(self, text: str) -> Dict:
        """Get LIME explanation for text quality"""
        try:
            def predict_fn(texts):
                X = self.vectorizer.transform(texts)
                return self.quality_classifier.predict_proba(X)
            
            explanation = self.lime_explainer.explain_instance(
                text, predict_fn, num_features=10, num_samples=500
            )
            
            return {
                'explanation_text': explanation.as_list(),
                'score': explanation.score,
                'local_prediction': explanation.local_pred
            }
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            return {}
    
    def _extract_function_features(self, func: Dict, file_info: Dict) -> Dict[str, float]:
        """Extract numerical features for function analysis"""
        features = {}
        
        # Basic features
        features['param_count'] = len(func.get('params', '').split(',')) if func.get('params') else 0
        features['has_docstring'] = 1.0 if func.get('docstring') else 0.0
        features['description_length'] = len(func.get('ai_description', ''))
        features['name_length'] = len(func['name'])
        
        # Semantic features
        description = func.get('ai_description', '').lower()
        features['has_verb'] = 1.0 if any(word in description for word in 
                                         ['calculates', 'processes', 'handles', 'manages', 'creates']) else 0.0
        features['has_specificity'] = 1.0 if any(word in description for word in 
                                                ['algorithm', 'validation', 'parsing', 'optimization']) else 0.0
        features['is_generic'] = 1.0 if any(phrase in description for phrase in 
                                           ['function that', 'method that', 'does something']) else 0.0
        
        return features
    
    def _extract_class_features(self, cls: Dict, file_info: Dict) -> Dict[str, float]:
        """Extract numerical features for class analysis"""
        features = {}
        
        features['has_docstring'] = 1.0 if cls.get('docstring') else 0.0
        features['description_length'] = len(cls.get('ai_description', ''))
        features['name_length'] = len(cls['name'])
        
        description = cls.get('ai_description', '').lower()
        features['has_purpose'] = 1.0 if any(word in description for word in 
                                            ['manages', 'handles', 'represents', 'implements']) else 0.0
        features['has_domain_context'] = 1.0 if any(word in description for word in 
                                                   ['user', 'data', 'file', 'network', 'database']) else 0.0
        
        return features
    
    def _extract_summary_features(self, file_info: Dict) -> Dict[str, float]:
        """Extract features for file summary analysis"""
        features = {}
        
        summary = file_info.get('ai_summary', '').lower()
        features['summary_length'] = len(summary)
        features['mentions_functions'] = 1.0 if 'function' in summary else 0.0
        features['mentions_classes'] = 1.0 if 'class' in summary else 0.0
        features['has_numbers'] = 1.0 if re.search(r'\d+', summary) else 0.0
        features['actual_function_count'] = len(file_info.get('functions', []))
        features['actual_class_count'] = len(file_info.get('classes', []))
        
        return features
    
    def _check_semantic_coherence(self, summary: str, file_info: Dict) -> float:
        """Check semantic coherence between summary and actual file content"""
        # Simple coherence check based on content matching
        functions = file_info.get('functions', [])
        classes = file_info.get('classes', [])
        
        coherence_score = 0.5  # Base score
        
        # Check if summary mentions correct counts
        function_count = len(functions)
        class_count = len(classes)
        
        if str(function_count) in summary:
            coherence_score += 0.2
        if str(class_count) in summary:
            coherence_score += 0.2
        
        # Check if summary mentions relevant keywords from function names
        all_names = [f['name'] for f in functions] + [c['name'] for c in classes]
        name_words = set()
        for name in all_names:
            # Convert camelCase and snake_case to words
            words = re.findall(r'[A-Z][a-z]*|[a-z]+|\d+', name)
            name_words.update([w.lower() for w in words])
        
        summary_words = set(summary.lower().split())
        overlap = len(name_words.intersection(summary_words))
        if name_words:
            coherence_score += 0.1 * min(overlap / len(name_words), 1.0)
        
        return min(coherence_score, 1.0)
    
    def _calculate_confidence(self, score: float, shap_values: np.ndarray) -> float:
        """Calculate confidence based on prediction certainty and explanation consistency"""
        # Confidence based on how far from 0.5 the score is (certainty)
        certainty = abs(score - 0.5) * 2
        
        # Explanation consistency (how concentrated the SHAP values are)
        if shap_values is not None and len(shap_values.shape) > 1:
            shap_variance = np.var(shap_values[0])
            explanation_consistency = min(shap_variance * 10, 1.0)  # Normalize
        else:
            explanation_consistency = 0.5
        
        return (certainty + explanation_consistency) / 2
    
    def _generate_function_recommendations(self, func: Dict, quality_score: float, 
                                         feature_importance: Dict, lime_explanation: Dict) -> List[str]:
        """Generate actionable recommendations for function documentation"""
        recommendations = []
        
        if quality_score < 0.6:
            recommendations.append("Consider rewriting the description to be more specific and actionable")
        
        if 'generic' in str(feature_importance.keys()).lower():
            recommendations.append("Avoid generic phrases like 'function that handles' or 'does something'")
        
        if len(func.get('ai_description', '')) < 20:
            recommendations.append("Expand the description to include what the function does, its parameters, and return value")
        
        # Check LIME explanation for negative words
        if lime_explanation and 'explanation_text' in lime_explanation:
            negative_words = [item[0] for item in lime_explanation['explanation_text'] if item[1] < 0]
            if negative_words:
                recommendations.append(f"Consider removing or rephrasing these words: {', '.join(negative_words[:3])}")
        
        if not func.get('docstring'):
            recommendations.append("Add a proper docstring to complement the AI description")
        
        return recommendations
    
    def _generate_class_recommendations(self, cls: Dict, quality_score: float,
                                      feature_importance: Dict, lime_explanation: Dict) -> List[str]:
        """Generate recommendations for class documentation"""
        recommendations = []
        
        if quality_score < 0.6:
            recommendations.append("Enhance class description with its purpose, main responsibilities, and usage context")
        
        if len(cls.get('ai_description', '')) < 25:
            recommendations.append("Expand description to explain what the class represents and its key methods")
        
        description = cls.get('ai_description', '').lower()
        if not any(word in description for word in ['manages', 'handles', 'represents', 'implements']):
            recommendations.append("Include action words that describe what the class does or represents")
        
        return recommendations
    
    def _generate_summary_recommendations(self, file_info: Dict, quality_score: float,
                                        feature_importance: Dict, lime_explanation: Dict) -> List[str]:
        """Generate recommendations for file summaries"""
        recommendations = []
        
        if quality_score < 0.6:
            recommendations.append("Rewrite file summary to better reflect the actual content and purpose")
        
        summary = file_info.get('ai_summary', '')
        if len(summary) < 30:
            recommendations.append("Expand summary to provide more context about the file's purpose and contents")
        
        # Check accuracy of counts
        actual_funcs = len(file_info.get('functions', []))
        actual_classes = len(file_info.get('classes', []))
        
        if actual_funcs > 0 and 'function' not in summary.lower():
            recommendations.append(f"Mention that the file contains {actual_funcs} functions")
        
        if actual_classes > 0 and 'class' not in summary.lower():
            recommendations.append(f"Mention that the file contains {actual_classes} classes")
        
        return recommendations
    
    def generate_xai_report(self, results: List[XAIValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive XAI validation report"""
        total_elements = len(results)
        avg_score = np.mean([r.score for r in results]) if results else 0.0
        avg_confidence = np.mean([r.confidence for r in results]) if results else 0.0
        
        # Categorize results
        high_quality = [r for r in results if r.score >= 0.8]
        medium_quality = [r for r in results if 0.6 <= r.score < 0.8]
        low_quality = [r for r in results if r.score < 0.6]
        
        # Most important features across all validations
        all_features = {}
        for result in results:
            for feature, importance in result.feature_importance.items():
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        
        avg_feature_importance = {
            feature: np.mean(importances) 
            for feature, importances in all_features.items()
        }
        
        return {
            'summary': {
                'total_elements': total_elements,
                'average_quality_score': round(avg_score, 3),
                'average_confidence': round(avg_confidence, 3),
                'high_quality_count': len(high_quality),
                'medium_quality_count': len(medium_quality),
                'low_quality_count': len(low_quality)
            },
            'feature_insights': {
                'most_important_features': dict(sorted(avg_feature_importance.items(), 
                                                     key=lambda x: abs(x[1]), reverse=True)[:10]),
                'positive_indicators': {k: v for k, v in avg_feature_importance.items() if v > 0},
                'negative_indicators': {k: v for k, v in avg_feature_importance.items() if v < 0}
            },
            'recommendations': {
                'global_recommendations': self._generate_global_recommendations(results),
                'priority_fixes': [r for r in results if r.score < 0.5]
            },
            'detailed_results': [
                {
                    'element_name': r.element_name,
                    'file_path': r.file_path,
                    'validation_type': r.validation_type,
                    'score': round(r.score, 3),
                    'confidence': round(r.confidence, 3),
                    'top_features': dict(list(r.explanation.get('top_features', {}).items())[:5]),
                    'recommendations': r.recommendations
                }
                for r in sorted(results, key=lambda x: x.score)
            ]
        }
    
    def _generate_global_recommendations(self, results: List[XAIValidationResult]) -> List[str]:
        """Generate global recommendations based on all validation results"""
        recommendations = []
        
        low_quality_count = len([r for r in results if r.score < 0.6])
        if low_quality_count > len(results) * 0.3:
            recommendations.append("Consider retraining or fine-tuning the AI model for better quality descriptions")
        
        avg_confidence = np.mean([r.confidence for r in results]) if results else 0
        if avg_confidence < 0.7:
            recommendations.append("Low confidence scores suggest reviewing AI-generated content more carefully")
        
        # Check for common issues
        generic_count = len([r for r in results if any('generic' in str(rec).lower() 
                                                      for rec in r.recommendations)])
        if generic_count > len(results) * 0.4:
            recommendations.append("Focus on training the AI to generate more specific, non-generic descriptions")
        
        return recommendations

# Example integration with the existing system
def create_xai_enhanced_generator():
    """Factory function to create XAI-enhanced documentation generator"""
    
    class XAIEnhancedDocsGenerator(DocsGenerator):
        def __init__(self):
            super().__init__()
            self.xai_validator = XAIDocumentationValidator()
        
        def generate_from_repo(self, repo_url, branch='main', use_ai_enhancement=True, use_xai_validation=True):
            """Enhanced generation with XAI validation"""
            success = super().generate_from_repo(repo_url, branch, use_ai_enhancement)
            
            if success and use_xai_validation:
                logger.info("Starting XAI validation...")
                
                # Run XAI validation
                xai_results = self.xai_validator.validate_with_xai(self.repo_info)
                xai_report = self.xai_validator.generate_xai_report(xai_results)
                
                # Save XAI reports
                repo_output_dir = os.path.join(self.output_dir, self.repo_info['name'])
                
                with open(os.path.join(repo_output_dir, 'xai_validation_report.json'), 'w') as f:
                    json.dump(xai_report, f, indent=2)
                
                # Generate XAI HTML report
                html_report = self._generate_xai_html_report(xai_report)
                with open(os.path.join(repo_output_dir, 'xai_validation_report.html'), 'w') as f:
                    f.write(html_report)
                
                logger.info(f"XAI validation completed. Average quality score: {xai_report['summary']['average_quality_score']}")
            
            return success
        
        def _generate_xai_html_report(self, xai_report: Dict) -> str:
            """Generate HTML report for XAI validation results"""
            # Implementation would create a comprehensive HTML report
            # showing SHAP/LIME explanations, feature importance, etc.
            pass
    
    return XAIEnhancedDocsGenerator

# Usage example
if __name__ == "__main__":
    # Test the XAI validator
    validator = XAIDocumentationValidator()
    
    # Mock data for testing
    test_repo_info = {
        'files': [{
            'name': 'test.py',
            'rel_path': 'test.py',
            'extension': '.py',
            'functions': [{
                'name': 'calculate_factorial',
                'params': 'n',
                'ai_description': 'Function that handles number calculations',
                'docstring': ''
            }],
            'classes': [{
                'name': 'DataProcessor',
                'ai_description': 'A class that handles data processing operations',
                'docstring': ''
            }],
            'ai_summary': 'Contains 1 class and 1 function.'
        }]
    }
    
    results = validator.validate_with_xai(test_repo_info)
    report = validator.generate_xai_report(results)
    
    print(f"XAI Validation Results:")
    print(f"Average Quality Score: {report['summary']['average_quality_score']}")
    print(f"Average Confidence: {report['summary']['average_confidence']}")
    print("\nTop Features:")
    for feature, importance in list(report['feature_insights']['most_important_features'].items())[:5]:
        print(f"  {feature}: {importance:.3f}")