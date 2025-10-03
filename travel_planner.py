import streamlit as st
from datetime import datetime, timedelta
import google.generativeai as genai
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure Streamlit page
st.set_page_config(
    page_title="Travel Itinerary Planner",
    page_icon="🌍",
    layout="wide"
)

def is_valid_gemini_key(key: str) -> bool:
    """Validate format of Gemini API key."""
    pattern = r"^AIza[0-9A-Za-z_\-]{35,41}$"
    return bool(re.fullmatch(pattern, key))

def setup_gemini_direct(api_key):
    if not api_key:
        return None, "No API key provided"
    api_key = api_key.strip()
    if not api_key.startswith('AIza'):
        return None, "Invalid API key format. Gemini API keys should start with 'AIza'"
    
    if len(api_key) < 34:        
        return None, "API key appears too short. Please check your API key"
        
    if not is_valid_gemini_key(api_key):
        return None, "Invalid API key format. It should start with 'AIza' and be 39–45 characters long."
    
    genai.configure(api_key=api_key)
    
    try:
        models = genai.list_models()
        available_models = [model.name for model in models if 'generateContent' in model.supported_generation_methods]
        
        if not available_models:
            return None, "No generation models available"
        
        model_name = 'gemini-pro'
        for model in available_models:
            if 'gemini-pro' in model:
                model_name = model
                break
        
        model = genai.GenerativeModel(model_name)
        return model, None
            
    except Exception as e:
        error_str = str(e)
        if "API_KEY" in error_str:
            return None, "Invalid API key"
        elif "quota" in error_str.lower():
            return None, "API quota exceeded"
        else:
            return None, f"Connection failed: {str(e)}"

# RAG System for Travel Data
class TravelRAGSystem:
    def __init__(self):
        self.knowledge_base = [
            "Flights are best for long distances and international travel",
            "Trains are comfortable for medium distances and scenic routes",
            "Buses are economical for short to medium distances",
            "Car rentals provide flexibility for local travel",
            "Metro systems are efficient in major cities for local transportation",
            "Taxis and ride-sharing are convenient for point-to-point travel",
            "Domestic flights in India: IndiGo, Air India, SpiceJet for quick travel between cities",
            "Indian Railways: Rajdhani Express for premium travel, Shatabdi for day journeys",
            "Local transport in metro cities: Delhi Metro, Mumbai Local trains, Bangalore BMTC buses",
            "Best areas to stay are usually city centers or near major attractions for convenience"
        ]
        
        self.vectorizer = TfidfVectorizer()
        self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_base)
    
    def search_travel_info(self, query, k=3):
        """Search for relevant travel information using TF-IDF similarity"""
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.knowledge_vectors).flatten()
            top_indices = similarities.argsort()[-k:][::-1]
            results = [self.knowledge_base[i] for i in top_indices if similarities[i] > 0.1]
            return "\n".join(results) if results else "General travel knowledge available"
        except:
            return "General travel knowledge available"

# Enhanced MakeMyTrip Data Scraper
class MakeMyTripScraper:
    @staticmethod
    def get_hotel_recommendations(destination):
        """Comprehensive hotel area recommendations for Indian destinations"""
        hotels_data = {
            "delhi": [
                "Connaught Place: Central location, excellent metro connectivity, business hotels",
                "Aerocity: Near airport, luxury hotels like JW Marriott, Holiday Inn",
                "Paharganj: Budget area near New Delhi Railway Station, backpacker hostels",
                "South Delhi: Hauz Khas, Greater Kailash - upscale residential areas",
                "Karol Bagh: Mid-range hotels, shopping area, good connectivity"
            ],
            "mumbai": [
                "South Mumbai: Marine Drive, Colaba, Nariman Point - premium locations near attractions",
                "Bandra: Suburban area with good connectivity, restaurants, and nightlife",
                "Andheri: Near airport, business hotels, good metro connectivity",
                "Juhu: Beach area, luxury resorts, family-friendly",
                "Powai: Lake area, business hotels, peaceful location"
            ],
            "goa": [
                "North Goa: Baga, Calangute, Candolim - beaches, nightlife, water sports",
                "South Goa: Palolem, Agonda, Colva - peaceful beaches, luxury resorts",
                "Panaji: Capital city, central location, heritage areas",
                "Old Goa: Heritage area, near churches and historical sites"
            ],
            "bangalore": [
                "MG Road: Central business district, luxury hotels, shopping",
                "Indiranagar: Trendy area with restaurants, pubs, and boutiques",
                "Whitefield: IT corridor, business hotels, good metro connectivity",
                "Koramangala: Residential area with good restaurants and connectivity"
            ],
            "jaipur": [
                "MI Road: Main street, central location, heritage hotels",
                "Bani Park: Peaceful area, heritage havelis converted to hotels",
                "Malviya Nagar: Modern area, good restaurants and shopping",
                "Amer Road: Near Amber Fort, luxury resorts"
            ],
            "kerala": [
                "Kochi: Fort Kochi heritage area, marine drive, city center",
                "Munnar: Hill station, tea garden resorts, central location",
                "Alleppey: Houseboat stays, beach area, canal side resorts",
                "Kovalam: Beach resorts, lighthouse area, cliff hotels"
            ]
        }
        
        destination_lower = destination.lower()
        for key in hotels_data:
            if key in destination_lower:
                return hotels_data[key]
        
        return ["City center area recommended for better connectivity to attractions"]

# Enhanced Travel Planner with RAG
class TravelPlanner:
    def __init__(self, model, api_key):
        self.model = model
        self.rag_system = TravelRAGSystem()
        self.knowledge = TravelKnowledge()
    
    def generate_itinerary(self, from_place, to_place, from_date, vacation_days):
        """Generate complete travel itinerary using RAG and knowledge base"""
        
        # Get RAG context for enhanced recommendations
        transport_context = self.rag_system.search_travel_info(f"best transportation from {from_place} to {to_place}")
        local_transport_context = self.rag_system.search_travel_info(f"local transportation in {to_place}")
        
        # Get enhanced hotel recommendations
        hotel_recommendations = MakeMyTripScraper.get_hotel_recommendations(to_place)
        
        # Get knowledge base recommendations
        distance_type = self.knowledge.estimate_distance(from_place, to_place)
        transport_advice = self.knowledge.get_transport_advice(from_place, to_place, distance_type)
        hotel_areas = self.knowledge.get_hotel_areas(to_place)
        
        # Combine RAG and knowledge base information
        combined_transport = f"{transport_context}\n\nAdditional Insights:\n{transport_advice}"
        combined_hotels = f"{chr(10).join(['- ' + area for area in hotel_recommendations])}\n\nKnowledge Base Recommendations:\n{chr(10).join(['- ' + area for area in hotel_areas])}"
        
        prompt = f"""
        Generate a detailed {vacation_days}-day travel itinerary from {from_place} to {to_place}.

        TRIP OVERVIEW:
        - From: {from_place}
        - To: {to_place}
        - Duration: {vacation_days} days
        - Start Date: {from_date}

        TRANSPORTATION ANALYSIS:
        {combined_transport}

        ACCOMMODATION STRATEGY:
        {combined_hotels}
        
        Local Transportation:
        {local_transport_context}
        
        ACCOMMODATION:
        Recommended areas: {', '.join(hotel_recommendations)}

        Please provide the itinerary in this comprehensive format:

        ===== TRAVEL ITINERARY =====
        
        TRIP OVERVIEW
        • Starting Point: {from_place}
        • Destination: {to_place}  
        • Duration: {vacation_days} days
        • Travel Period: {from_date}

        TRANSPORTATION GUIDE
        [Provide specific transportation options based on the analysis above]

        ACCOMMODATION STRATEGY  
        [Recommend where to stay with specific area details]
        
        Local Transportation:
        [Recommended transportation methods within the destination]

        DAILY ITINERARY
        Day 1: Arrival and Initial Exploration
        • Morning: [Specific activities]
        • Afternoon: [Specific activities]
        • Evening: [Specific activities]

        Day 2 to Day {vacation_days-1}: Main Sightseeing
        [Detailed daily plans with specific attractions, restaurants, and timing]

        Day {vacation_days}: Departure
        • Morning: [Final activities]
        • Afternoon: [Departure preparations]

        TRAVEL TIPS
        • Budget Planning: [Cost estimates and money-saving tips and provide Total cost]
        • Local Customs & Etiquette: [Important cultural notes]
        • Safety Advice: [Safety precautions and emergency contacts]
        • Packing Guide: [Seasonal packing recommendations]
        • Booking Tips: [When and how to book transportation and accommodation]

        ===== ITINERARY END =====

        Make it practical, specific to Indian travel context, and include actual sightseeing spots, restaurants, and activities.
        Provide specific recommendations for transportation booking websites, hotel booking platforms, and local apps.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._format_response(response.text)
        except Exception as e:
            return f"Error generating itinerary: {str(e)}"
    
    def _format_response(self, response):
        """Ensure consistent formatting of LLM response"""
        # Clean markdown formatting
        cleaned_response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)
        cleaned_response = re.sub(r'\*(.*?)\*', r'\1', cleaned_response)
        cleaned_response = re.sub(r'#+\s*', '', cleaned_response)
        
        return cleaned_response

# Travel Knowledge Base (Enhanced with RAG integration)
class TravelKnowledge:
    def __init__(self):
        self.transport_knowledge = {
            "long_distance": {
                "flights": "Best for distances over 500km. Fastest option. Airlines: IndiGo, Air India, SpiceJet",
                "trains": "Good for 200-500km. Comfortable: Rajdhani, Shatabdi. Budget: Express trains",
                "buses": "Economical for under 200km. Operators: RedBus, VRL Travels"
            },
            "local_transport": {
                "metro_cities": "Use metro systems in Delhi, Mumbai, Bangalore, Chennai",
                "taxis": "Ola, Uber for convenient point-to-point travel",
                "auto_rickshaws": "Good for short distances, negotiate fare first",
                "buses": "City bus services available in all major cities"
            }
        }
        
        self.hotel_recommendations = {
            "delhi": [
                "Connaught Place: Central location, metro connectivity, business hotels",
                "Aerocity: Near airport, luxury hotels (JW Marriott, Holiday Inn)",
                "Paharganj: Budget area near railway station, hostels",
                "South Delhi: Hauz Khas, GK - upscale residential areas"
            ],
            "mumbai": [
                "South Mumbai: Marine Drive, Colaba - premium locations near attractions",
                "Bandra: Suburban with good connectivity, restaurants, nightlife",
                "Andheri: Near airport, business hotels, metro connectivity"
            ],
            "goa": [
                "North Goa: Baga, Calangute - beaches, nightlife, water sports",
                "South Goa: Palolem, Agonda - peaceful beaches, luxury resorts",
                "Panaji: Capital city, central location, heritage areas"
            ],
            "bangalore": [
                "MG Road: Central business district, luxury hotels, shopping",
                "Indiranagar: Trendy area with restaurants, pubs, boutiques",
                "Whitefield: IT corridor, business hotels"
            ],
            "jaipur": [
                "MI Road: Main street, central location, heritage hotels",
                "Bani Park: Peaceful area, heritage havelis converted to hotels"
            ],
            "kerala": [
                "Kochi: Fort Kochi heritage area, marine drive",
                "Munnar: Hill station, tea garden resorts",
                "Alleppey: Houseboat stays, beach area"
            ],
            "default": [
                "City center area recommended for better connectivity",
                "Near major transportation hubs for convenience",
                "Tourist areas for easier access to attractions"
            ]
        }
    
    def get_transport_advice(self, from_place, to_place, distance_estimate):
        advice = []
        
        if distance_estimate == "long":
            advice.append("FLIGHTS: Best option for quick travel")
            advice.append("   - Check: IndiGo, Air India, SpiceJet")
            advice.append("   - Book in advance for better prices")
            advice.append("TRAINS: Comfortable alternative")
            advice.append("   - Premium: Rajdhani Express")
            advice.append("   - Day journey: Shatabdi Express")
        elif distance_estimate == "medium":
            advice.append("TRAINS: Recommended for comfort and scenery")
            advice.append("BUSES: Economical option, overnight journeys available")
        else:
            advice.append("ROAD: Car rental or taxi recommended")
            advice.append("BUS: Frequent services available")
        
        advice.append("LOCAL TRANSPORT:")
        advice.append("   - Metro: Available in major cities")
        advice.append("   - Taxis: Ola, Uber for convenience")
        advice.append("   - Auto-rickshaws: For short distances")
        
        return "\n".join(advice)
    
    def get_hotel_areas(self, destination):
        dest_lower = destination.lower()
        for city in self.hotel_recommendations:
            if city in dest_lower:
                return self.hotel_recommendations[city]
        return self.hotel_recommendations["default"]
    
    def estimate_distance(self, from_place, to_place):
        from_lower = from_place.lower()
        to_lower = to_place.lower()
        
        if from_lower == to_lower:
            return "short"
        
        long_distance_pairs = [
            {"delhi", "mumbai"}, {"delhi", "chennai"}, {"delhi", "bangalore"},
            {"delhi", "kolkata"}, {"delhi", "hyderabad"}, {"mumbai", "chennai"},
            {"mumbai", "kolkata"}, {"bangalore", "delhi"}, {"chennai", "delhi"}
        ]
        
        pair = {from_lower, to_lower}
        if any(pair == long_pair for long_pair in long_distance_pairs):
            return "long"
        
        if any(city in from_lower and any(other in to_lower for other in ["goa", "pune", "jaipur"]) for city in ["mumbai", "delhi"]):
            return "medium"
        
        return "short"

def main():
    st.title("AI Travel Itinerary Planner")
    st.markdown("Plan your perfect Indian vacation with AI-powered recommendations")
    
    # Initialize session state safely
    if 'ai_configured' not in st.session_state:
        st.session_state.ai_configured = False
    if 'planner' not in st.session_state:
        st.session_state.planner = None
    if 'config_error' not in st.session_state:
        st.session_state.config_error = None
    if 'reset_trigger' not in st.session_state:
        st.session_state.reset_trigger = 0
    if 'show_reset' not in st.session_state:
        st.session_state.show_reset = False

    # Sidebar for API key
    with st.sidebar:
        st.header("AI Configuration")
        
        # Use a unique key that changes on reset to clear the input
        api_key_input_key = f"api_key_input_{st.session_state.reset_trigger}"
        
        api_key = st.text_input("Gemini API Key", type="password", 
                               help="Enter your Google Gemini API key",
                               key=api_key_input_key,
                               value="")
        
        # Configure AI button - always visible
        configure_clicked = st.button("Configure AI", 
                                    use_container_width=True)
        
        # Show configuration status
        if st.session_state.ai_configured:
            st.success("AI Ready")
        
        # Handle configuration - this must be processed BEFORE showing errors
        config_processed = False
        if configure_clicked:
            if not api_key:
                st.session_state.config_error = "Please enter an API key"
                st.session_state.ai_configured = False
                st.session_state.show_reset = True
            else:
                try:
                    model, error = setup_gemini_direct(api_key)
                    if error:
                        st.session_state.config_error = error
                        st.session_state.ai_configured = False
                        st.session_state.show_reset = True
                    else:
                        st.session_state.planner = TravelPlanner(model, api_key)
                        st.session_state.ai_configured = True
                        st.session_state.config_error = None
                        st.session_state.show_reset = True
                except Exception as e:
                    st.session_state.config_error = f"Configuration failed: {str(e)}"
                    st.session_state.ai_configured = False
                    st.session_state.show_reset = True
            config_processed = True
        
        # Show error immediately if exists (AFTER processing configuration)
        if st.session_state.config_error:
            st.error(st.session_state.config_error)
        
        # Show Reset Configuration button ONLY if show_reset is True
        if st.session_state.show_reset:
            reset_clicked = st.button("Reset Configuration", 
                                    use_container_width=True,
                                    key="reset_button")
            
            if reset_clicked:
                # Reset all states
                st.session_state.ai_configured = False
                st.session_state.planner = None
                st.session_state.config_error = None
                st.session_state.show_reset = False
                st.session_state.reset_trigger += 1  # Change the key to clear input
                st.rerun()  # Force immediate refresh
        
        st.markdown("---")
        st.info("""
        **How to use:**
        1. Enter your Gemini API key
        2. Click **Configure AI**
        3. Fill in trip details
        4. Generate itinerary
        """)
    
    # Main input form
    if st.session_state.ai_configured:
        st.header("Enter Trip Details")
        
        with st.form("travel_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                from_place = st.text_input("Departure City", 
                                         placeholder="e.g., Delhi", 
                                         value="Delhi")
                to_place = st.text_input("Destination City", 
                                       placeholder="e.g., Goa", 
                                       value="Goa")
            
            with col2:
                from_date = st.date_input("Start Date", 
                                        datetime.now() + timedelta(days=7))
                vacation_days = st.number_input("Number of Days", 
                                              min_value=1, 
                                              max_value=30, 
                                              value=5)
            
            travel_style = st.selectbox("Travel Style", 
                                      ["Budget", "Comfort", "Luxury", "Adventure", "Family"])
            
            submitted = st.form_submit_button("Generate Travel Plan")
        
        if submitted:
            if not from_place or not to_place:
                st.error("Please fill in all required fields")
            else:
                try:
                    itinerary = st.session_state.planner.generate_itinerary(
                        from_place, 
                        to_place, 
                        from_date.strftime("%B %d, %Y"), 
                        vacation_days
                    )
                    
                    # Display results
                    st.success("✨ Your Travel Itinerary is Ready!")
                    st.markdown("---")
                    
                    # Display itinerary in expandable sections
                    st.subheader("📅 Your Travel Plan")
                    
                    with st.expander("View Full Itinerary", expanded=True):
                        st.text_area("Itinerary Details", itinerary, height=500, label_visibility="collapsed")
                    
                    # Quick tips
                    st.subheader("💡 Quick Tips")
                    tip_col1, tip_col2, tip_col3 = st.columns(3)
                    
                    with tip_col1:
                        st.info("**Booking**\n\nBook flights/trains 2-3 weeks in advance for best prices")
                    
                    with tip_col2:
                        st.info("**Packing**\n\nCheck weather forecast and pack accordingly")
                    
                    with tip_col3:
                        st.info("**Local Travel**\n\nDownload Ola/Uber apps for convenient transport")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Itinerary",
                        data=itinerary,
                        file_name=f"travel_plan_{to_place}_{vacation_days}days.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating itinerary: {str(e)}")
    
    else:
        # Show instructions if AI is not configured
        st.header("Get Started")
        st.info("""
        **To begin planning your trip:**
        
        1. **Enter your Gemini API key** in the sidebar
        2. **Click the 'Configure AI' button** to initialize the AI system
        3. **Fill in your travel details** once the AI is ready
        
        You only need to configure the AI once per session!
        """)
        
        # Example itinerary preview
        with st.expander("📋 See Example Itinerary Format"):
            st.markdown("""
            **TRIP OVERVIEW**
            - From: Delhi
            - To: Goa  
            - Duration: 5 days
            - Start Date: December 25, 2024
            
            **TRANSPORTATION GUIDE**
            - Flight: IndiGo, SpiceJet from Delhi to Goa
            - Local: Taxis, rental scooters, local buses
            
            **ACCOMMODATION STRATEGY**
            - North Goa: Baga, Calangute for beaches and nightlife
            - South Goa: Palolem for peaceful resorts
            
            **DAILY ITINERARY**
            Day 1: Arrival and Beach Exploration
            - Morning: Flight arrival, check-in
            - Afternoon: Calangute Beach visit
            - Evening: Beachside dinner at Baga
            
            ...and more detailed daily plans!
            """)

if __name__ == "__main__":
    main()