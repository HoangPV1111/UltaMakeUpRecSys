import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from ortools.linear_solver import pywraplp
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# --- ENHANCED COMPUTER VISION WITH IMPROVED ACNE DETECTION ---
st.set_page_config(page_title="Ulta Makeup Recommendations", layout="wide")
st.title(' Ulta Makeup Recommendations ')

class ImprovedSkinAnalyzer:
    """Enhanced skin analysis with better acne detection algorithms"""
    
    def __init__(self):
        self.setup_models()
        
    def setup_models(self):
        """Initialize detection models"""
        try:
            # For now, we'll use improved traditional CV methods
            # You can later integrate trained YOLO models here
            self.models_available = True
        except Exception as e:
            st.warning(f"Advanced models not available: {e}")
            self.models_available = False
    
    def detect_skin_tone(self, image):
        """Enhanced skin tone detection using multiple regions and color analysis"""
        # Convert to different color spaces for better analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        h, w, _ = image.shape
        
        # Define facial regions (avoiding areas with potential acne/conditions)
        regions = [
            (w//2-20, h//2+20, 40, 30),     # Lower center (below nose)
            (w//4, h//3, 30, 30),           # Left cheek
            (3*w//4, h//3, 30, 30),         # Right cheek
            (w//2, h//4, 20, 20),           # Forehead center
        ]
        
        skin_colors = []
        for x, y, width, height in regions:
            if 0 <= x < w-width and 0 <= y < h-height:
                # Use LAB color space for better skin tone analysis
                patch_lab = lab[y:y+height, x:x+width]
                patch_rgb = image[y:y+height, x:x+width]
                
                # Filter out potential acne/red areas
                patch_hsv = hsv[y:y+height, x:x+width]
                
                # Create mask to exclude red/inflamed areas
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 50, 50])
                upper_red2 = np.array([180, 255, 255])
                
                mask1 = cv2.inRange(patch_hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(patch_hsv, lower_red2, upper_red2)
                red_mask = mask1 + mask2
                
                # Use only non-red pixels for skin tone analysis
                non_red_mask = cv2.bitwise_not(red_mask)
                if np.sum(non_red_mask) > 0:
                    masked_patch = cv2.bitwise_and(patch_rgb, patch_rgb, mask=non_red_mask)
                    non_zero_pixels = masked_patch[non_red_mask > 0]
                    if len(non_zero_pixels) > 0:
                        avg_color = np.mean(non_zero_pixels.reshape(-1, 3), axis=0)
                        skin_colors.append(avg_color)
        
        if skin_colors:
            final_skin_color = np.mean(skin_colors, axis=0)
            
            # Enhanced classification using multiple factors
            brightness = np.mean(final_skin_color)
            
            # Convert to HSV for additional analysis
            hsv_color = cv2.cvtColor(np.uint8([[final_skin_color]]), cv2.COLOR_RGB2HSV)[0][0]
            
            # Classification with more granular categories
            if brightness < 70:
                return 'Deep', final_skin_color
            elif brightness < 100:
                return 'Medium-Deep', final_skin_color
            elif brightness < 140:
                return 'Medium', final_skin_color
            elif brightness < 180:
                return 'Medium-Light', final_skin_color
            else:
                return 'Light', final_skin_color
        
        return 'Medium', [128, 128, 128]
    
    def detect_acne_improved(self, image):
        """Improved acne detection using multiple methods"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        h, w = gray.shape
        acne_locations = []
        
        # Method 1: Red color detection (inflammatory acne)
        # Define red ranges in HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours for red areas
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 200:  # Filter by size
                # Get bounding rectangle
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Check if it's roughly circular (acne-like)
                aspect_ratio = float(w_rect) / h_rect
                if 0.5 < aspect_ratio < 2.0:
                    center_x = x + w_rect // 2
                    center_y = y + h_rect // 2
                    radius = max(w_rect, h_rect) // 2
                    
                    acne_locations.append({
                        'x': center_x, 'y': center_y, 'r': radius,
                        'type': 'inflammatory', 'confidence': 0.8
                    })
        
        # Method 2: Dark spot detection (blackheads, comedones)
        # Use LAB color space for better detection
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        
        # Detect dark spots
        mean_l = np.mean(l_channel)
        dark_threshold = mean_l - 20
        dark_mask = l_channel < dark_threshold
        
        # Additional filtering using A channel (green-red)
        # Blackheads often have different color characteristics
        kernel_small = np.ones((2,2), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
        
        # Find dark spot contours
        contours_dark, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_dark:
            area = cv2.contourArea(contour)
            if 3 < area < 50:  # Smaller than inflammatory acne
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                aspect_ratio = float(w_rect) / h_rect
                if 0.4 < aspect_ratio < 2.5:
                    center_x = x + w_rect // 2
                    center_y = y + h_rect // 2
                    radius = max(w_rect, h_rect) // 2
                    
                    acne_locations.append({
                        'x': center_x, 'y': center_y, 'r': radius,
                        'type': 'comedonal', 'confidence': 0.6
                    })
        
        # Method 3: Texture analysis for subtle acne
        # Apply Gaussian blur and find differences
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blurred)
        
        # Threshold the difference
        _, texture_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        # Find texture-based acne
        contours_texture, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_texture:
            area = cv2.contourArea(contour)
            if 8 < area < 100:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                center_x = x + w_rect // 2
                center_y = y + h_rect // 2
                radius = max(w_rect, h_rect) // 2
                
                # Check if not already detected
                is_duplicate = False
                for existing in acne_locations:
                    dist = np.sqrt((center_x - existing['x'])**2 + (center_y - existing['y'])**2)
                    if dist < 10:  # Too close to existing detection
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    acne_locations.append({
                        'x': center_x, 'y': center_y, 'r': radius,
                        'type': 'textural', 'confidence': 0.5
                    })
        
        # Calculate severity
        acne_count = len(acne_locations)
        
        # Weight by confidence and type
        weighted_count = sum([
            spot['confidence'] * (1.5 if spot['type'] == 'inflammatory' else 1.0)
            for spot in acne_locations
        ])
        
        if weighted_count > 15:
            severity = "Severe"
        elif weighted_count > 8:
            severity = "Moderate"
        elif weighted_count > 2:
            severity = "Mild"
        else:
            severity = "Clear"
        
        return acne_count, severity, acne_locations, [red_mask, dark_mask, texture_mask]
    
    def detect_skin_conditions_comprehensive(self, image):
        """Comprehensive skin condition analysis"""
        results = {}
        
        # Skin tone
        skin_tone, skin_color = self.detect_skin_tone(image)
        results['skin_tone'] = (skin_tone, skin_color)
        
        # Enhanced acne detection
        acne_count, acne_severity, acne_spots, acne_masks = self.detect_acne_improved(image)
        results['acne'] = (acne_count, acne_severity, acne_spots, acne_masks)
        
        # Quick wrinkle detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=15, maxLineGap=8)
        
        wrinkle_count = len(lines) if lines is not None else 0
        wrinkle_severity = "Severe" if wrinkle_count > 20 else "Moderate" if wrinkle_count > 10 else "Mild" if wrinkle_count > 5 else "None"
        results['wrinkles'] = (wrinkle_count, wrinkle_severity, lines)
        
        # Pigmentation analysis
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        std_lightness = np.std(l_channel)
        mean_lightness = np.mean(l_channel)
        
        # Detect pigmentation irregularities
        pigment_threshold = mean_lightness - 1.2 * std_lightness
        pigment_mask = l_channel < pigment_threshold
        coverage = (np.sum(pigment_mask) / pigment_mask.size) * 100
        
        pig_severity = "Severe" if coverage > 12 else "Moderate" if coverage > 6 else "Mild" if coverage > 2 else "None"
        results['pigmentation'] = (coverage, pig_severity, pigment_mask)
        
        return results

# Initialize the enhanced analyzer
@st.cache_resource
def get_enhanced_analyzer():
    return ImprovedSkinAnalyzer()

analyzer = get_enhanced_analyzer()

# Load recommendation models and data
@st.cache_resource
def load_recommendation_models():
    # Load the model
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    product_data = pd.read_csv('reddit_product_embeddings.csv')
    # Giữ nguyên product_data để sau còn dùng merge
    product_copy = product_data.copy()
    # Xóa cột trong bản copy
    product_copy = product_copy.drop(columns=['overall_product_rating'], errors='ignore')

    
    # Load product info
    product_info = pd.read_csv('cleaned_makeup_products.csv')
    product_info = product_info[['product_link_id', 'product_name', 'brand', 'price', 'category']].copy()
    product_info.rename(columns={'category': 'category_name'}, inplace=True)
    
    return model, product_copy, product_info

model, product_data, product_info = load_recommendation_models()

def generate_recommendations(analysis_results, user_prefs, max_budget):
    """Generate personalized product recommendations using the recommendation system"""
    # Create user profile based on analysis and preferences
    user_profile = {
        'professional_review': 0, 
        'vibe_review': 0, 
        'redness_review': 1 if analysis_results['acne'][1] in ['Severe', 'Moderate'] else 0,
        'dry_review': 1 if user_prefs['skin_type'] in ['Dry', 'Sensitive'] else 0,
        'light_coverage_review': 1 if user_prefs['coverage_pref'] == 'Light' else 0,
        'medium_coverage_review': 1 if user_prefs['coverage_pref'] == 'Medium' else 0,
        'full_coverage_review': 1 if user_prefs['coverage_pref'] == 'Full' else 0,
        'young_review': 1 if user_prefs['age_range'] in ['Under 20', '20-30'] else 0,
        'mother_review': 1 if user_prefs['age_range'] in ['30-40', '40-50', '50+'] else 0,
        'skin_concerns_review': 1 if analysis_results['acne'][1] != 'Clear' or analysis_results['pigmentation'][1] != 'None' else 0,
        'white_review': 1 if analysis_results['skin_tone'][0] in ['Light', 'Medium-Light'] else 0,
        'tan_review': 1 if analysis_results['skin_tone'][0] in ['Medium', 'Medium-Deep'] else 0,
        'black_review': 1 if analysis_results['skin_tone'][0] == 'Deep' else 0,
        'acne_review': 1 if analysis_results['acne'][1] != 'Clear' else 0,
        'comfortable_wear_review': 1 if user_prefs['activity_level'] in ['High Activity', 'Professional'] else 0,
        'easy_use_review': 1 if user_prefs['experience_level'] == 'Beginner' else 0,
        'wrinkles_review': 1 if analysis_results['wrinkles'][1] in ['Moderate', 'Severe'] else 0
    }
    
    # Prepare product data
    product_copy = product_data.copy()
    le = LabelEncoder()
    product_copy['category'] = le.fit_transform(product_copy['category'])
    product_copy = product_copy.fillna(product_copy.mean())
    # Add user profile to product data
    for col, val in user_profile.items(): 
        product_copy[col] = val
    
    # Reorder the DataFrame to match the model's expected feature order
    feature_names = model.feature_names_in_
    product_copy = product_copy[feature_names]
    
    # Make predictions
    predictions = model.predict_proba(product_copy)
    product_copy['predicted_score'] = predictions[:, 0] + 2*predictions[:, 1] + 3*predictions[:, 2] + 4*predictions[:, 3] + 5*predictions[:, 4]
    product_copy['product_link_id'] = product_data['product_link_id']
    
    # Merge with product info
    product_budget = pd.merge(product_copy, product_info, how='inner', on='product_link_id')
    product_budget['category_name'].fillna(value='Uncategorized', inplace=True)
    
    # Determine allowed categories based on user profile
    skin_categories = ['Foundation', 'Tinted Moisturizer', 'BB & CC Creams']
    allowed_skin_categories = []
    allowed_makeup_products = []
    
    if user_profile['light_coverage_review'] == 1: 
        allowed_makeup_products = ['Blush', 'Concealer', 'Makeup Remover', 'Setting Spray & Powder']
        if user_profile['easy_use_review'] == 1:
            allowed_skin_categories = ['Foundation', 'Tinted Moisturizer']
        elif user_profile['acne_review'] == 1:
            allowed_skin_categories = ['Foundation', 'BB & CC Creams']
        elif user_profile['comfortable_wear_review'] == 1:
            allowed_skin_categories = ['Tinted Moisturizer', 'BB & CC Creams']
        else: 
            allowed_skin_categories = ['Foundation', 'Tinted Moisturizer', 'BB & CC Creams']
    elif user_profile['medium_coverage_review'] == 1: 
        allowed_makeup_products = ['Face Primer', 'Blush', 'Bronzer', 'Concealer', 'Makeup Remover', 'Setting Spray & Powder']
        allowed_skin_categories = ['Foundation', 'BB & CC Creams']
    elif user_profile['full_coverage_review']: 
        allowed_makeup_products = ['Face Primer', 'Blush', 'Bronzer', 'Contouring', 'Highlighter', 
                                  'Color Correcting', 'Concealer', 'Makeup Remover', 'Setting Spray & Powder']
        allowed_skin_categories = ['Foundation']
    
    # Optimization setup
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Solver not created.")
    
    # Create variables
    product_vars = {}
    for i, row in product_budget.iterrows():
        product_vars[i] = solver.BoolVar(f'product_{row["product_link_id"]}')
    
    # Budget constraint
    budget_constraint = solver.Constraint(0, max_budget)
    for i, row in product_budget.iterrows():
        budget_constraint.SetCoefficient(product_vars[i], row['price'])
    
    # Category constraints
    allowed_skin_constraint = solver.Constraint(0, 1)
    disallowed_constraint = solver.Constraint(0, 0)
    
    for cat in allowed_skin_categories:
        cat_df = product_budget[product_budget['category_name'] == cat]
        for i, row in cat_df.iterrows():
            allowed_skin_constraint.SetCoefficient(product_vars[i], 1)
    
    for cat in skin_categories:
        if cat in allowed_skin_categories: continue
        cat_df = product_budget[product_budget['category_name'] == cat]
        for i, row in cat_df.iterrows():
            disallowed_constraint.SetCoefficient(product_vars[i], 1)
    
    for cat in product_budget['category_name'].unique():
        if cat in allowed_makeup_products: continue
        if cat in allowed_skin_categories: continue
        cat_df = product_budget[product_budget['category_name'] == cat]
        for i, row in cat_df.iterrows():
            disallowed_constraint.SetCoefficient(product_vars[i], 1)
    
    for cat in product_budget['category_name'].unique():
        if cat in skin_categories: continue
        if cat not in allowed_makeup_products: continue
        cat_constraint = solver.Constraint(0, 1)
        cat_df = product_budget[product_budget['category_name'] == cat]
        for i, row in cat_df.iterrows():
            cat_constraint.SetCoefficient(product_vars[i], 1)
    
    # Objective function
    objective = solver.Objective()
    for i, row in product_budget.iterrows():
        objective.SetCoefficient(product_vars[i], row['predicted_score'])
    objective.SetMaximization()
    
    # Solve the problem
    status = solver.Solve()
    
    # Collect results
    recommendations = []
    if status == pywraplp.Solver.OPTIMAL:
        for i, row in product_budget.iterrows():
            if product_vars[i].solution_value() == 1:
                recommendations.append({
                    'category': row['category_name'],
                    'product': row['product_name'],
                    'brand': row['brand'],
                    'price': row['price'],
                    'score': row['predicted_score'],
                    'reason': get_recommendation_reason(row['category_name'], analysis_results, user_prefs)
                })
    
    return recommendations

def get_recommendation_reason(category, analysis_results, user_prefs):
    """Generate a reason for recommending a specific product category"""
    reasons = {
        'Foundation': f"Matched your skin tone ({analysis_results['skin_tone'][0]}) and coverage preference ({user_prefs['coverage_pref']})",
        'Concealer': f"Targeted coverage for {analysis_results['acne'][1].lower()} acne and pigmentation",
        'Color Correcting': f"Helps neutralize {analysis_results['pigmentation'][1].lower()} pigmentation",
        'Setting Spray & Powder': f"Locks in makeup for your {user_prefs['activity_level'].lower()} lifestyle",
        'Face Primer': f"Prepares skin and improves makeup longevity for {user_prefs['skin_type'].lower()} skin",
        'BB & CC Creams': "Lightweight coverage with skincare benefits for everyday wear",
        'Tinted Moisturizer': "Sheer coverage with hydration for a natural look",
        'Blush': "Adds healthy glow to complement your skin tone",
        'Bronzer': "Enhances natural warmth for your complexion",
        'Makeup Remover': "Gentle cleansing for your skincare routine"
    }
    return reasons.get(category, "Recommended based on your profile and preferences")

# --- STREAMLIT INTERFACE ---
st.markdown("""
<div style='background: linear-gradient(90deg, #ff6b6b, #ffa726, #42a5f5); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: white; text-align: center; margin: 0;'>
         Ulta Makeup Recommendations
    </h3>
</div>
""", unsafe_allow_html=True)

# Main upload section
st.subheader(' Upload Your Image for Comprehensive Analysis')

uploaded_file = st.file_uploader(
    "Choose a clear, well-lit photo of your face",
    type=['jpg', 'jpeg', 'png'],
    help="For best results: good lighting, front-facing, no makeup, clean skin visible"
)

if uploaded_file is not None:
    # Load and process image
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    # Create two columns for image and quick results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Your uploaded image', use_container_width=True)
    
    with col2:
        with st.spinner(" Analyzing your skin... This may take a moment."):
            # Perform comprehensive analysis
            analysis_results = analyzer.detect_skin_conditions_comprehensive(image_np)
            
        # Display quick results
        st.subheader(" Quick Analysis Results")
        
        # Metrics in a nice layout
        metric_cols = st.columns(2)
        
        with metric_cols[0]:
            skin_tone = analysis_results['skin_tone'][0]
            st.metric(" Skin Tone", skin_tone)
            
            acne_count, acne_severity, _, _ = analysis_results['acne']
            color = "🔴" if acne_severity in ["Severe", "Moderate"] else "🟡" if acne_severity == "Mild" else "🟢"
            st.metric(f"{color} Acne Level", acne_severity, f"{acne_count} spots detected")
        
        with metric_cols[1]:
            wrinkle_count, wrinkle_severity, _ = analysis_results['wrinkles']
            color = "🔴" if wrinkle_severity in ["Severe", "Moderate"] else "🟡" if wrinkle_severity == "Mild" else "🟢"
            st.metric(f"{color} Aging Signs", wrinkle_severity, f"{wrinkle_count} lines")
            
            coverage, pig_severity, _ = analysis_results['pigmentation']
            color = "🔴" if pig_severity in ["Severe", "Moderate"] else "🟡" if pig_severity == "Mild" else "🟢"
            st.metric(f"{color} Pigmentation", pig_severity, f"{coverage:.1f}% coverage")
    
    # Quick preferences
    st.divider()
    st.subheader(" Quick Preferences")
    
    pref_col1, pref_col2, pref_col3 = st.columns(3)
    
    with pref_col1:
        age_range = st.selectbox('Age Range', ['Under 20', '20-30', '30-40', '40-50', '50+'])
        experience_level = st.selectbox('Makeup Experience', ['Beginner', 'Intermediate', 'Advanced'])
    
    with pref_col2:
        skin_type = st.selectbox('Skin Type', ['Oily', 'Dry', 'Combination', 'Normal', 'Sensitive'])
        max_budget = st.number_input('Max Budget per Product ($)', min_value=5, max_value=200, value=30)
    
    with pref_col3:
        coverage_pref = st.selectbox('Coverage Preference', ['Light', 'Medium', 'Full', 'Auto-detect'])
        activity_level = st.selectbox('Lifestyle', ['Low Activity', 'Moderate', 'High Activity', 'Professional'])
    
    # Auto-detect coverage if selected
    if coverage_pref == 'Auto-detect':
        acne_severity = analysis_results['acne'][1]
        pig_severity = analysis_results['pigmentation'][1]
        
        if acne_severity in ['Severe', 'Moderate'] or pig_severity in ['Severe', 'Moderate']:
            auto_coverage = 'Full'
        elif acne_severity == 'Mild' or pig_severity == 'Mild':
            auto_coverage = 'Medium'
        else:
            auto_coverage = 'Light'
    else:
        auto_coverage = coverage_pref
    
    # Generate recommendations
    if st.button(" Generate Smart Recommendations", type="primary", use_container_width=True):
        with st.spinner("Creating personalized recommendations..."):
            user_prefs = {
                'age_range': age_range,
                'experience_level': experience_level,
                'skin_type': skin_type,
                'coverage_pref': auto_coverage,
                'activity_level': activity_level
            }
            
            recommendations = generate_recommendations(analysis_results, user_prefs, max_budget)
            
            if recommendations:
                st.success(f" Found {len(recommendations)} personalized recommendations for you!")
                st.subheader(" Your Personalized Product Recommendations")
                
                for rec in recommendations:
                    with st.expander(f"{rec['category']}: {rec['product']} by {rec['brand']} (${rec['price']:.2f})", expanded=True):
                        st.write(f"**Why recommended:** {rec['reason']}")
                        st.write(f"**Predicted match score:** {rec['score']:.2f}")
                        st.write(f"**Price:** ${rec['price']:.2f}")
                        
                        if st.button("Find Similar Products", key=f"find_{rec['product']}"):
                            st.info(" Searching for similar products...")
                
                # Analysis summary
                st.divider()
                st.subheader(" Complete Analysis Summary")
                
                summary_cols = st.columns(3)
                
                with summary_cols[0]:
                    st.info(f"""
                    **Skin Profile**
                    - Tone: {analysis_results['skin_tone'][0]}
                    - Type: {skin_type}
                    - Age Range: {age_range}
                    """)
                
                with summary_cols[1]:
                    st.warning(f"""
                    **Detected Conditions**
                    - Acne: {analysis_results['acne'][1]} ({analysis_results['acne'][0]} spots)
                    - Aging: {analysis_results['wrinkles'][1]}
                    - Pigmentation: {analysis_results['pigmentation'][1]}
                    """)
                
                with summary_cols[2]:
                    total_price = sum(rec['price'] for rec in recommendations)
                    st.success(f"""
                    **Recommendations**
                    - Coverage: {auto_coverage}
                    - Products: {len(recommendations)}
                    - Total Budget: ${total_price:.2f}
                    """)
            else:
                st.warning("No recommendations found that match your criteria. Try adjusting your preferences.")

else:
    # Instructions when no image is uploaded
    st.info("""
     **Upload your image to get started!**
    
    """)
