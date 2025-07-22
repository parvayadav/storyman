# AI Multi-Agent Creative Engine - Streamlit App
# Install: pip install streamlit requests pillow openai beautifulsoup4 opencv-python numpy plotly

import streamlit as st
import requests
import os
import json
import time
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import openai
from dataclasses import dataclass
from typing import List, Dict, Any
import base64
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import zipfile

# Page config
st.set_page_config(
    page_title="AI Creative Engine",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class BrandConfig:
    primary_color: str = "#FF6B35"
    secondary_color: str = "#004E89" 
    accent_color: str = "#FFFFFF"
    font_family: str = "Arial"
    brand_name: str = "YourBrand"

@dataclass
class CreativeRequest:
    prompt: str
    dimensions: tuple
    platform: str = "general"
    style: str = "modern"

class WebCrawlerAgent:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.visited_urls = set()
        self.found_images = []
        
    def crawl_website(self, max_pages: int = 10, progress_callback=None):
        """Crawl website and extract product images with progress updates"""
        
        urls_to_visit = [self.base_url]
        visited_count = 0
        
        while urls_to_visit and visited_count < max_pages:
            url = urls_to_visit.pop(0)
            if url in self.visited_urls:
                continue
                
            if progress_callback:
                progress_callback(f"Crawling: {url[:50]}...", visited_count / max_pages)
                
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.visited_urls.add(url)
                    visited_count += 1
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract images
                    images = soup.find_all('img')
                    for img in images:
                        img_url = img.get('src') or img.get('data-src')
                        if img_url:
                            full_url = urljoin(url, img_url)
                            if self._is_product_image(img, full_url):
                                self.found_images.append({
                                    'url': full_url,
                                    'alt': img.get('alt', ''),
                                    'page_url': url,
                                    'context': self._get_image_context(img)
                                })
                    
                    # Find new URLs to crawl
                    links = soup.find_all('a', href=True)
                    for link in links[:5]:
                        new_url = urljoin(url, link['href'])
                        if self._should_crawl_url(new_url):
                            urls_to_visit.append(new_url)
                            
                time.sleep(0.5)  # Be respectful
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error crawling {url}: {e}", visited_count / max_pages)
                
        return self.found_images
    
    def _is_product_image(self, img_tag, img_url: str) -> bool:
        """Determine if image is likely a product image"""
        if not any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
            return False
            
        width = img_tag.get('width')
        height = img_tag.get('height')
        if width and height:
            try:
                w, h = int(width), int(height)
                if w < 100 or h < 100:
                    return False
            except:
                pass
                
        alt_text = img_tag.get('alt', '').lower()
        class_name = img_tag.get('class', [])
        parent_classes = []
        
        if img_tag.parent:
            parent_classes = img_tag.parent.get('class', [])
            
        product_indicators = ['product', 'item', 'catalog', 'shop', 'buy']
        exclude_indicators = ['logo', 'icon', 'banner', 'ad', 'social']
        
        context = f"{alt_text} {' '.join(class_name)} {' '.join(parent_classes)}".lower()
        
        has_product_indicator = any(indicator in context for indicator in product_indicators)
        has_exclude_indicator = any(indicator in context for indicator in exclude_indicators)
        
        return has_product_indicator and not has_exclude_indicator
    
    def _get_image_context(self, img_tag) -> str:
        context = ""
        if img_tag.parent:
            context = img_tag.parent.get_text(strip=True)[:200]
        return context
    
    def _should_crawl_url(self, url: str) -> bool:
        parsed = urlparse(url)
        base_domain = urlparse(self.base_url).netloc
        
        if parsed.netloc != base_domain:
            return False
            
        skip_paths = ['/admin', '/login', '/cart', '/checkout', '/account']
        return not any(skip in url.lower() for skip in skip_paths)

class ImageClassificationAgent:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def classify_images(self, image_list: List[Dict], progress_callback=None) -> List[Dict]:
        """Classify and sort images using OpenAI Vision"""
        
        classified_images = []
        total_images = min(len(image_list), 10)  # Limit for demo
        
        for i, img_data in enumerate(image_list[:total_images]):
            if progress_callback:
                progress_callback(f"Classifying image {i+1}/{total_images}", i / total_images)
                
            try:
                response = requests.get(img_data['url'], timeout=10)
                if response.status_code == 200:
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
                    classification = self._classify_single_image(image_base64, img_data)
                    img_data.update(classification)
                    classified_images.append(img_data)
                    
            except Exception as e:
                st.error(f"Error classifying image {i+1}: {e}")
                
        return classified_images
    
    def _classify_single_image(self, image_base64: str, img_data: Dict) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Analyze this product image and return a JSON response with:
                                - category: main product category
                                - subcategory: specific type
                                - colors: dominant colors (list)
                                - style: visual style
                                - quality_score: 1-10 rating
                                - brand_elements: any visible brand elements
                                - composition: description of layout
                                
                                Context: {img_data.get('context', '')}
                                Alt text: {img_data.get('alt', '')}"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            else:
                return {"category": "unknown", "quality_score": 5}
                
        except Exception as e:
            return {"category": "unknown", "quality_score": 5, "error": str(e)}

class CreativeAdaptationAgent:
    def __init__(self, api_key: str, brand_config: BrandConfig):
        self.client = openai.OpenAI(api_key=api_key)
        self.brand = brand_config
        
    def adapt_images(self, classified_images: List[Dict], creative_request: CreativeRequest, progress_callback=None) -> List[Dict]:
        """Adapt images based on requirements"""
        
        adapted_images = []
        total_images = min(len(classified_images), 5)
        
        for i, img_data in enumerate(classified_images[:total_images]):
            if progress_callback:
                progress_callback(f"Adapting image {i+1}/{total_images}", i / total_images)
                
            try:
                response = requests.get(img_data['url'], timeout=10)
                if response.status_code == 200:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        tmp_file.write(response.content)
                        original_path = tmp_file.name
                    
                    adaptations = self._create_adaptations(original_path, img_data, creative_request)
                    img_data['adaptations'] = adaptations
                    adapted_images.append(img_data)
                    
                    # Clean up
                    os.unlink(original_path)
                    
            except Exception as e:
                st.error(f"Error adapting image {i+1}: {e}")
                
        return adapted_images
    
    def _create_adaptations(self, image_path: str, img_data: Dict, request: CreativeRequest) -> List[Dict]:
        """Create multiple adaptations of an image"""
        adaptations = []
        
        img = Image.open(image_path)
        
        # 1. Basic resize and crop
        resized = self._resize_and_crop(img, request.dimensions)
        adaptations.append({
            "type": "resized",
            "image": resized,
            "description": f"Resized to {request.dimensions}"
        })
        
        # 2. Add brand elements
        branded = self._add_brand_elements(resized.copy(), request)
        adaptations.append({
            "type": "branded",
            "image": branded,
            "description": f"Added brand elements"
        })
        
        # 3. Color adjustment
        color_adjusted = self._adjust_colors(resized.copy())
        adaptations.append({
            "type": "color_adjusted",
            "image": color_adjusted,
            "description": f"Color adjusted for brand"
        })
        
        return adaptations
    
    def _resize_and_crop(self, img: Image.Image, target_dims: tuple) -> Image.Image:
        target_w, target_h = target_dims
        original_w, original_h = img.size
        
        target_ratio = target_w / target_h
        original_ratio = original_w / original_h
        
        if original_ratio > target_ratio:
            new_h = target_h
            new_w = int(target_h * original_ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            left = (new_w - target_w) // 2
            img = img.crop((left, 0, left + target_w, target_h))
        else:
            new_w = target_w
            new_h = int(target_w / original_ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            top = (new_h - target_h) // 2
            img = img.crop((0, top, target_w, top + target_h))
            
        return img
    
    def _add_brand_elements(self, img: Image.Image, request: CreativeRequest) -> Image.Image:
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        text = request.prompt[:50] if len(request.prompt) > 0 else self.brand.brand_name
        
        try:
            font = ImageFont.truetype("arial.ttf", size=max(20, width//20))
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = height - text_height - 20
        
        padding = 10
        draw.rectangle([
            x - padding, 
            y - padding, 
            x + text_width + padding, 
            y + text_height + padding
        ], fill=self.brand.primary_color)
        
        draw.text((x, y), text, fill=self.brand.accent_color, font=font)
        
        return img
    
    def _adjust_colors(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(enhanced)

def main():
    st.title("🎨 AI Multi-Agent Creative Engine")
    st.markdown("*Crawl, classify, and adapt product images automatically*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key")
        
        st.subheader("🎯 Brand Configuration")
        brand_name = st.text_input("Brand Name", value="YourBrand")
        primary_color = st.color_picker("Primary Color", "#FF6B35")
        secondary_color = st.color_picker("Secondary Color", "#004E89")
        accent_color = st.color_picker("Accent Color", "#FFFFFF")
        
        st.subheader("📐 Creative Request")
        prompt = st.text_area("Creative Prompt", "Summer sale - bright and energetic style")
        
        platform = st.selectbox("Platform", ["Instagram Square", "Facebook Post", "LinkedIn Post", "Custom"])
        
        if platform == "Instagram Square":
            dimensions = (1080, 1080)
        elif platform == "Facebook Post":
            dimensions = (1200, 630)
        elif platform == "LinkedIn Post":
            dimensions = (1200, 627)
        else:
            col1, col2 = st.columns(2)
            width = col1.number_input("Width", value=1080, min_value=100)
            height = col2.number_input("Height", value=1080, min_value=100)
            dimensions = (int(width), int(height))
        
        st.metric("Target Dimensions", f"{dimensions[0]} x {dimensions[1]}")
        
        style = st.selectbox("Style", ["modern", "vintage", "minimal", "bold", "elegant"])
        
        max_pages = st.slider("Max Pages to Crawl", 1, 20, 5)
    
    # Main interface
    st.header("🌐 Website Crawler")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        website_url = st.text_input("Website URL", placeholder="https://your-website.com")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        start_processing = st.button("🚀 Start Processing", type="primary")
    
    if start_processing and website_url and api_key:
        # Create brand config
        brand_config = BrandConfig(
            primary_color=primary_color,
            secondary_color=secondary_color,
            accent_color=accent_color,
            brand_name=brand_name
        )
        
        # Create creative request
        creative_request = CreativeRequest(
            prompt=prompt,
            dimensions=dimensions,
            platform=platform.lower(),
            style=style
        )
        
        # Initialize agents
        crawler = WebCrawlerAgent(website_url)
        classifier = ImageClassificationAgent(api_key)
        adapter = CreativeAdaptationAgent(api_key, brand_config)
        
        # Processing steps with progress
        st.header("🔄 Processing Pipeline")
        
        # Step 1: Crawling
        st.subheader("1. 🕷️ Crawling Website")
        crawl_progress = st.progress(0)
        crawl_status = st.empty()
        
        def crawl_callback(message, progress):
            crawl_status.text(message)
            crawl_progress.progress(progress)
        
        with st.spinner("Crawling website for product images..."):
            found_images = crawler.crawl_website(max_pages, crawl_callback)
        
        if not found_images:
            st.error("❌ No product images found. Try a different website or check the URL.")
            return
        
        st.success(f"✅ Found {len(found_images)} product images")
        
        # Display found images
        with st.expander("🖼️ View Found Images"):
            cols = st.columns(4)
            for i, img_data in enumerate(found_images[:8]):  # Show first 8
                with cols[i % 4]:
                    try:
                        st.image(img_data['url'], caption=img_data.get('alt', 'Product Image')[:30], width=150)
                    except:
                        st.text("Image preview unavailable")
        
        # Step 2: Classification
        st.subheader("2. 🔍 Classifying Images")
        classify_progress = st.progress(0)
        classify_status = st.empty()
        
        def classify_callback(message, progress):
            classify_status.text(message)
            classify_progress.progress(progress)
        
        with st.spinner("Analyzing images with AI..."):
            classified_images = classifier.classify_images(found_images, classify_callback)
        
        if not classified_images:
            st.error("❌ No images could be classified. Check your API key and try again.")
            return
        
        st.success(f"✅ Classified {len(classified_images)} images")
        
        # Show classification results
        if classified_images:
            categories = {}
            quality_scores = []
            
            for img in classified_images:
                category = img.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                quality_scores.append(img.get('quality_score', 5))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Category distribution
                if categories:
                    fig = px.pie(
                        values=list(categories.values()),
                        names=list(categories.keys()),
                        title="Product Categories Found"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Quality scores
                if quality_scores:
                    fig = go.Figure(data=[go.Histogram(x=quality_scores, nbinsx=10)])
                    fig.update_layout(title="Image Quality Distribution", xaxis_title="Quality Score", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Step 3: Adaptation
        st.subheader("3. 🎨 Adapting Images")
        adapt_progress = st.progress(0)
        adapt_status = st.empty()
        
        def adapt_callback(message, progress):
            adapt_status.text(message)
            adapt_progress.progress(progress)
        
        with st.spinner("Creating brand-adapted versions..."):
            adapted_images = adapter.adapt_images(classified_images, creative_request, adapt_callback)
        
        st.success(f"✅ Created adaptations for {len(adapted_images)} images")
        
        # Display results
        st.header("🎯 Results")
        
        total_adaptations = sum(len(img.get('adaptations', [])) for img in adapted_images)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Images Found", len(found_images))
        col2.metric("Images Processed", len(adapted_images))
        col3.metric("Adaptations Created", total_adaptations)
        col4.metric("Avg Quality Score", f"{np.mean(quality_scores):.1f}/10" if quality_scores else "N/A")
        
        # Show adapted images
        for i, img_data in enumerate(adapted_images):
            with st.expander(f"📸 Image {i+1}: {img_data.get('category', 'Unknown')} (Quality: {img_data.get('quality_score', 'N/A')})"):
                
                # Original image
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original:**")
                    try:
                        st.image(img_data['url'], width=300)
                    except:
                        st.text("Original image unavailable")
                    
                    st.json({
                        "Category": img_data.get('category', 'Unknown'),
                        "Quality Score": img_data.get('quality_score', 'N/A'),
                        "Colors": img_data.get('colors', []),
                        "Style": img_data.get('style', 'Unknown')
                    })
                
                with col2:
                    st.markdown("**Adaptations:**")
                    
                    # Show adaptations
                    adaptations = img_data.get('adaptations', [])
                    if adaptations:
                        tabs = st.tabs([f"{adapt['type'].title()}" for adapt in adaptations])
                        
                        for j, (tab, adaptation) in enumerate(zip(tabs, adaptations)):
                            with tab:
                                st.image(adaptation['image'], width=300, caption=adaptation['description'])
                                
                                # Download button for each adaptation
                                img_buffer = io.BytesIO()
                                adaptation['image'].save(img_buffer, format='PNG')
                                st.download_button(
                                    f"📥 Download {adaptation['type'].title()}",
                                    data=img_buffer.getvalue(),
                                    file_name=f"adapted_{adaptation['type']}_{i}_{j}.png",
                                    mime="image/png",
                                    key=f"download_{i}_{j}"
                                )
        
        # Download all as ZIP
        if adapted_images:
            st.header("📦 Download All")
            
            # Create ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for i, img_data in enumerate(adapted_images):
                    for j, adaptation in enumerate(img_data.get('adaptations', [])):
                        img_buffer = io.BytesIO()
                        adaptation['image'].save(img_buffer, format='PNG')
                        zip_file.writestr(f"adapted_{adaptation['type']}_{i}_{j}.png", img_buffer.getvalue())
            
            st.download_button(
                "📥 Download All Adaptations (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"ai_creative_adaptations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
    
    elif start_processing:
        if not website_url:
            st.error("❌ Please enter a website URL")
        if not api_key:
            st.error("❌ Please enter your OpenAI API key")
    
    # Footer
    st.markdown("---")
    st.markdown("🤖 **AI Multi-Agent Creative Engine** - Built with Streamlit")
    
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **🕷️ Web Crawler**: Systematically discovers product images on your website
        2. **🔍 AI Classification**: Uses GPT-4 Vision to analyze and categorize images
        3. **🎨 Smart Adaptation**: Creates brand-aligned variations with proper dimensions
        4. **📊 Analytics**: Provides insights on image quality and categories
        5. **📥 Export**: Download individual images or bulk ZIP file
        
        **Tips:**
        - Start with e-commerce sites for best results
        - Ensure your OpenAI API key has GPT-4 Vision access
        - Experiment with different prompts and brand colors
        - Use the quality scores to identify best images for campaigns
        """)

if __name__ == "__main__":
    main()
