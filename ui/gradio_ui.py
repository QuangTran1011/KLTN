import gradio as gr
import requests
import json
import pandas as pd
from google.cloud import bigquery
from update_realtime_feat import push_new_item_sequence
from loguru import logger

# Cấu hình API endpoint
API_BASE_URL = "http://kltn.recsys.com"

# Cấu hình BigQuery
BQ_PROJECT_ID = "turing-thought-481409-d8"  
BQ_DATASET = "kltn"  
BQ_TABLE = "item_meta_data"  

# URL ảnh mặc định
DEFAULT_IMAGE_URL = "https://via.placeholder.com/150x200?text=No+Image"

# Store để lưu trạng thái liked items của mỗi user
liked_items_store = {}

def get_item_details_from_bigquery(item_ids):
    """Lấy thông tin chi tiết sản phẩm từ BigQuery"""
    if not item_ids:
        return {}
    
    try:
        client = bigquery.Client(project=BQ_PROJECT_ID)
        
        # Tạo query với list item_ids
        item_ids_str = "', '".join(item_ids)
        query = f"""
        SELECT 
            parent_asin,
            title,
            price,
            image
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE parent_asin IN ('{item_ids_str}')
        """
        
        results = client.query(query).result()
        
        # Chuyển thành dictionary
        item_details = {}
        for row in results:
            item_details[row.parent_asin] = {
                'title': row.title or 'N/A',
                'price': row.price or 'N/A',
                'image': row.image if row.image else DEFAULT_IMAGE_URL
            }
        
        return item_details
    
    except Exception as e:
        print(f"Error querying BigQuery: {e}")
        return {}

def create_product_card_html(rank, item_id, score, details, is_liked=False):
    """Tạo HTML cho product card"""
    title = details.get('title', 'N/A')
    price = details.get('price', 'N/A')
    image = details.get('image', DEFAULT_IMAGE_URL)
    
    price_display = f"${price}" if price != 'N/A' else 'N/A'
    
    liked_badge = """
        <div style="margin-top: 10px; padding: 8px; background: #e8f5e9; border-radius: 6px; 
                    color: #2e7d32; font-weight: bold; text-align: center;">
            ❤️ Already Liked
        </div>
    """ if is_liked else ""
    
    card_html = f"""
    <div class="product-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; 
                margin-bottom: 15px; display: flex; gap: 15px; background: white; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s;">
        <div style="flex-shrink: 0;">
            <img src="{image}" alt="{title}" 
                 style="width: 120px; height: 160px; object-fit: cover; border-radius: 4px;"
                 onerror="this.src='{DEFAULT_IMAGE_URL}'">
        </div>
        <div style="flex-grow: 1;">
            <div style="font-size: 18px; font-weight: bold; color: #1a1a1a; margin-bottom: 8px;">
                #{rank} - {title[:80]}{"..." if len(title) > 80 else ""}
            </div>
            <div style="margin-bottom: 5px;">
                <span style="color: #666;">Item ID:</span> 
                <span style="font-family: monospace; background: #f0f0f0; padding: 2px 6px; border-radius: 3px;">{item_id}</span>
            </div>
            <div style="margin-bottom: 5px;">
                <span style="color: #666;">Price:</span> 
                <span style="font-weight: bold; color: #B12704; font-size: 18px;">{price_display}</span>
            </div>
            <div style="margin-bottom: 10px;">
                <span style="color: #666;">Score:</span> 
                <span style="background: #007185; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;">
                    {score:.4f}
                </span>
            </div>
            {liked_badge}
        </div>
    </div>
    """
    return card_html

def get_recommendations(user_id: str, top_k_retrieval: int, count: int):
    """Gọi API User-to-Item với Rerank và lấy thông tin từ BigQuery"""
    if not user_id:
        return "⚠️ Vui lòng nhập User ID", [], pd.DataFrame()
    
    try:
        # Gọi API recommendations
        response = requests.get(
            f"{API_BASE_URL}/recs/u2i/rerank",
            params={
                "user_id": user_id,
                "top_k_retrieval": top_k_retrieval,
                "count": count,
                "debug": False
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Thông tin user
        info_text = f"""
### 👤 User Information
- **User ID:** {data.get('user_id', 'N/A')}
- **Total Recommendations:** {len(data.get('recommendations', {}).get('rec_item_ids', []))}
"""
        
        # Lấy item sequence nếu có
        if "features" in data and "item_sequence" in data["features"]:
            seq = data["features"]["item_sequence"]
            info_text += f"- **Recent Items:** {', '.join(seq[:5])}" + (f"... (+{len(seq)-5} more)" if len(seq) > 5 else "")
        
        # Hiển thị liked items
        if user_id in liked_items_store and liked_items_store[user_id]:
            liked_count = len(liked_items_store[user_id])
            info_text += f"\n- **❤️ Liked Items (This Session):** {liked_count} items"
            info_text += f"\n  - Latest: {', '.join(liked_items_store[user_id][-5:])}"
        
        # Lấy recommendations
        product_data = []
        df_data = []
        
        if "recommendations" in data:
            recs = data["recommendations"]
            item_ids = recs.get("rec_item_ids", [])
            scores = recs.get("rec_scores", [])
            
            # Lấy chi tiết từ BigQuery
            item_details = get_item_details_from_bigquery(item_ids)
            
            # Lấy danh sách liked items của user
            user_liked_items = liked_items_store.get(user_id, [])
            
            # Tạo data cho mỗi sản phẩm
            for rank, (item_id, score) in enumerate(zip(item_ids, scores), 1):
                details = item_details.get(item_id, {
                    'title': 'N/A',
                    'price': 'N/A',
                    'image': DEFAULT_IMAGE_URL
                })
                is_liked = item_id in user_liked_items
                
                product_data.append({
                    'rank': rank,
                    'item_id': item_id,
                    'score': score,
                    'details': details,
                    'is_liked': is_liked
                })
                
                # Data cho DataFrame
                df_data.append({
                    "Rank": rank,
                    "Item ID": item_id,
                    "Title": details.get('title', 'N/A')[:50] + "..." if len(details.get('title', '')) > 50 else details.get('title', 'N/A'),
                    "Price": details.get('price', 'N/A'),
                    "Score": round(score, 4),
                    "Status": "❤️ Liked" if is_liked else "Available"
                })
            
            df = pd.DataFrame(df_data)
        else:
            df = pd.DataFrame()
        
        return info_text, product_data, df
        
    except requests.exceptions.Timeout:
        return "⏰ Request timeout - API không phản hồi", [], pd.DataFrame()
    except requests.exceptions.ConnectionError:
        return f"❌ Không thể kết nối tới API: {API_BASE_URL}", [], pd.DataFrame()
    except requests.exceptions.HTTPError as e:
        return f"❌ HTTP Error: {e.response.status_code} - {e.response.text}", [], pd.DataFrame()
    except Exception as e:
        return f"❌ Error: {str(e)}", [], pd.DataFrame()

def handle_like_item(item_id, user_id, products):
    """Xử lý like item và cập nhật UI"""
    try:
        # Lưu vào store tạm thời
        if user_id not in liked_items_store:
            liked_items_store[user_id] = []
        if item_id not in liked_items_store[user_id]:
            liked_items_store[user_id].append(item_id)
            logger.info(f"User {user_id} liked item {item_id} (pending sync)")
        
        # Cập nhật trạng thái liked trong products
        updated_products = []
        for product in products:
            if product['item_id'] == item_id:
                product['is_liked'] = True
            updated_products.append(product)
        
        liked_count = len(liked_items_store[user_id])
        status = f"✅ Liked **{item_id}**! Total liked: **{liked_count}** items. Click 'Reload Recommendations' to update."
        
        return updated_products, status
        
    except Exception as e:
        logger.error(f"Error in like item: {e}")
        return products, f"❌ Error: {str(e)}"

def sync_likes_and_refresh(user_id, top_k, count):
    """Sync tất cả liked items và refresh recommendations"""
    try:
        # Lấy danh sách items đã like
        liked_items = liked_items_store.get(user_id, [])
        
        if not liked_items:
            # Vẫn refresh nhưng báo không có items mới
            info, products, df = get_recommendations(user_id, top_k, count)
            status = "⚠️ No new items liked. Showing current recommendations."
            return info, products, df, status
        
        # Push tất cả items vào online store
        logger.info(f"Syncing {len(liked_items)} liked items for user {user_id}")
        push_new_item_sequence(user_id=user_id, new_items=liked_items, sequence_length=10)
        
        # Refresh recommendations
        info, products, df = get_recommendations(user_id, top_k, count)
        
        status = f"✅ Synced **{len(liked_items)}** liked items and refreshed recommendations!"
        
        return info, products, df, status
        
    except Exception as e:
        logger.error(f"Error in sync and refresh: {e}")
        # Vẫn cố gắng load recommendations dù có lỗi
        info, products, df = get_recommendations(user_id, top_k, count)
        return info, products, df, f"⚠️ Error syncing: {str(e)}"

# Tạo Gradio Interface
with gr.Blocks(title="Recommendation System") as demo:
    gr.Markdown("""
    # 🎯 Real-time Recommendation System
    ### Hệ thống gợi ý sản phẩm với Like Real-time và Manual Refresh
    """)
    
    # Configuration Section
    with gr.Accordion("⚙️ Configuration", open=False):
        with gr.Row():
            api_url_input = gr.Textbox(
                label="🔗 API URL",
                value=API_BASE_URL,
                placeholder="http://localhost:8000"
            )
        with gr.Row():
            bq_project = gr.Textbox(label="BigQuery Project ID", value=BQ_PROJECT_ID)
            bq_dataset = gr.Textbox(label="BigQuery Dataset", value=BQ_DATASET)
            bq_table = gr.Textbox(label="BigQuery Table", value=BQ_TABLE)
        
        update_config_btn = gr.Button("💾 Update Configuration")
        config_status = gr.Markdown()
        
        def update_config(api_url, project, dataset, table):
            global API_BASE_URL, BQ_PROJECT_ID, BQ_DATASET, BQ_TABLE
            API_BASE_URL = api_url.rstrip('/')
            BQ_PROJECT_ID = project
            BQ_DATASET = dataset
            BQ_TABLE = table
            return f"✅ Configuration updated successfully!"
        
        update_config_btn.click(
            update_config,
            inputs=[api_url_input, bq_project, bq_dataset, bq_table],
            outputs=[config_status]
        )
    
    # Main Input Section
    with gr.Row():
        with gr.Column(scale=2):
            user_id_input = gr.Textbox(
                label="👤 User ID",
                placeholder="Nhập User ID của bạn...",
                lines=1
            )
    
    with gr.Row():
        top_k_slider = gr.Slider(
            minimum=10,
            maximum=500,
            value=100,
            step=10,
            label="📊 Top K Retrieval"
        )
        count_slider = gr.Slider(
            minimum=1,
            maximum=50,
            value=10,
            step=1,
            label="🎯 Number of Recommendations"
        )
    
    submit_btn = gr.Button("🚀 Get Recommendations", variant="primary", size="lg")
    
    gr.Markdown("---")
    
    # Results Section
    info_output = gr.Markdown(label="Information")
    like_status = gr.Markdown()
    
    # State để lưu products data
    products_state = gr.State([])
    
    gr.Markdown("""
    ### 📚 Recommended Products
    💡 **Hướng dẫn:** 
    1. Click nút ❤️ Like trên nhiều sản phẩm bạn thích
    2. Sau đó bấm nút **'🔄 Reload Recommendations'** để cập nhật gợi ý mới
    """)
    
    # Reload button
    with gr.Row():
        reload_btn = gr.Button("🔄 Reload Recommendations (Sync All Likes)", variant="primary", size="lg")
    
    # Container cho products - sử dụng dynamic components
    @gr.render(inputs=[products_state, user_id_input])
    def render_products(products, user_id):
        """Render products với like buttons"""
        if not products:
            gr.Markdown("_No products to display. Click 'Get Recommendations' first._")
            return
        
        for product in products:
            item_id = product['item_id']
            
            with gr.Row():
                with gr.Column(scale=10):
                    gr.HTML(create_product_card_html(
                        product['rank'],
                        product['item_id'],
                        product['score'],
                        product['details'],
                        product['is_liked']
                    ))
                
                with gr.Column(scale=2, min_width=120):
                    if product['is_liked']:
                        # Nút đỏ khi đã like
                        gr.Button(
                            f"❤️ Liked #{product['rank']}", 
                            variant="stop",  # Màu đỏ
                            size="sm",
                            interactive=False
                        )
                    else:
                        # Nút trắng khi chưa like
                        like_btn = gr.Button(
                            f"🤍 Like #{product['rank']}", 
                            variant="secondary",  # Màu trắng
                            size="sm"
                        )
                        
                        # Tạo function wrapper để bind đúng item_id
                        def create_like_handler(curr_item_id):
                            def handler(uid, prods):
                                return handle_like_item(curr_item_id, uid, prods)
                            return handler
                        
                        # Bind click event
                        like_btn.click(
                            fn=create_like_handler(item_id),
                            inputs=[user_id_input, products_state],
                            outputs=[products_state, like_status]
                        )
    
    with gr.Accordion("📊 Summary Table", open=True):
        results_table = gr.DataFrame(
            label="Recommendations Summary",
            wrap=True
        )
    
    # Examples
    gr.Markdown("### 💡 Ví dụ:")
    gr.Examples(
        examples=[
            ["user_123", 100, 10],
            ["user_456", 150, 20],
            ["user_789", 200, 15],
        ],
        inputs=[user_id_input, top_k_slider, count_slider],
    )
    
    # Submit button action
    submit_btn.click(
        get_recommendations,
        inputs=[user_id_input, top_k_slider, count_slider],
        outputs=[info_output, products_state, results_table]
    ).then(
        lambda: "",
        outputs=[like_status]
    )
    
    # Reload button action - SYNC VÀ REFRESH
    reload_btn.click(
        sync_likes_and_refresh,
        inputs=[user_id_input, top_k_slider, count_slider],
        outputs=[info_output, products_state, results_table, like_status]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )