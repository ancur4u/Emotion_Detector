# Convert msambare FER2013 Dataset to CSV Format
# Specifically designed for https://www.kaggle.com/datasets/msambare/fer2013

import os
import cv2
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

def convert_msambare_dataset(dataset_path):
    """
    Convert msambare FER2013 dataset to CSV format
    Expected structure:
    fer2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
    """
    
    # Emotion mapping (matches the folder names exactly)
    emotion_mapping = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    data = []
    total_processed = 0
    
    # Process both train and test folders
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        
        if not os.path.exists(split_path):
            st.warning(f"Folder not found: {split_path}")
            continue
            
        st.write(f"Processing {split} folder...")
        
        # Process each emotion folder
        for emotion_folder in emotion_mapping.keys():
            emotion_path = os.path.join(split_path, emotion_folder)
            
            if not os.path.exists(emotion_path):
                st.warning(f"Emotion folder not found: {emotion_path}")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(emotion_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            st.write(f"  {emotion_folder}: {len(image_files)} images")
            
            # Process each image
            for i, img_file in enumerate(image_files):
                if i % 500 == 0:  # Update progress every 500 images
                    st.write(f"    Processed {i}/{len(image_files)} from {emotion_folder}")
                
                try:
                    img_path = os.path.join(emotion_path, img_file)
                    
                    # Load image as grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        # Try with PIL if OpenCV fails
                        pil_img = Image.open(img_path).convert('L')
                        img = np.array(pil_img)
                    
                    if img is None:
                        continue
                    
                    # Resize to 48x48 (standard FER2013 size)
                    img = cv2.resize(img, (48, 48))
                    
                    # Convert to pixel string (FER2013 format)
                    pixels = ' '.join([str(pixel) for pixel in img.flatten()])
                    
                    # Determine usage (train vs test)
                    usage = 'Training' if split == 'train' else 'PublicTest'
                    
                    data.append({
                        'emotion': emotion_mapping[emotion_folder],
                        'pixels': pixels,
                        'Usage': usage
                    })
                    
                    total_processed += 1
                    
                except Exception as e:
                    st.write(f"Error processing {img_file}: {e}")
                    continue
    
    st.success(f"✅ Total images processed: {total_processed}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    st.title("📁 FER2013 msambare Dataset Converter")
    st.write("Convert the msambare FER2013 dataset to CSV format")
    
    st.markdown("""
    ## 📋 Instructions for msambare FER2013 Dataset:
    
    1. **Extract your downloaded ZIP** from Kaggle
    2. **Find the main folder** (usually called 'fer2013' or similar)
    3. **Enter the path** to that folder below
    4. **Click Convert** to create the CSV
    
    ### Expected Folder Structure:
    ```
    your-path/fer2013/
    ├── train/
    │   ├── angry/     (images)
    │   ├── disgust/   (images)
    │   ├── fear/      (images)
    │   ├── happy/     (images)
    │   ├── neutral/   (images)
    │   ├── sad/       (images)
    │   └── surprise/  (images)
    └── test/
        ├── angry/     (images)
        ├── disgust/   (images)
        ├── fear/      (images)
        ├── happy/     (images)
        ├── neutral/   (images)
        ├── sad/       (images)
        └── surprise/  (images)
    ```
    """)
    
    # Path input with example
    dataset_path = st.text_input(
        "Enter path to your fer2013 dataset folder:",
        placeholder="C:/Users/YourName/Downloads/fer2013",
        help="This should be the folder containing 'train' and 'test' subfolders"
    )
    
    if dataset_path and os.path.exists(dataset_path):
        st.success(f"✅ Found folder: {dataset_path}")
        
        # Verify structure
        train_path = os.path.join(dataset_path, 'train')
        test_path = os.path.join(dataset_path, 'test')
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            st.success("✅ Found both 'train' and 'test' folders!")
            
            # Show emotion folder counts
            st.subheader("📊 Dataset Overview")
            
            emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            overview_data = []
            
            for emotion in emotions:
                train_emotion_path = os.path.join(train_path, emotion)
                test_emotion_path = os.path.join(test_path, emotion)
                
                train_count = 0
                test_count = 0
                
                if os.path.exists(train_emotion_path):
                    train_count = len([f for f in os.listdir(train_emotion_path) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                if os.path.exists(test_emotion_path):
                    test_count = len([f for f in os.listdir(test_emotion_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                overview_data.append({
                    'Emotion': emotion.capitalize(),
                    'Train Images': train_count,
                    'Test Images': test_count,
                    'Total': train_count + test_count
                })
            
            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df)
            
            total_images = overview_df['Total'].sum()
            st.info(f"📈 Total images to convert: {total_images}")
            
            # Convert button
            if st.button("🔄 Convert to CSV", type="primary"):
                with st.spinner("Converting dataset... This may take several minutes."):
                    df = convert_msambare_dataset(dataset_path)
                    
                    if len(df) > 0:
                        st.success(f"✅ Successfully converted {len(df)} images!")
                        
                        # Save CSV
                        csv_path = os.path.join(dataset_path, "fer2013_converted.csv")
                        df.to_csv(csv_path, index=False)
                        st.success(f"💾 Saved as: {csv_path}")
                        
                        # Show final summary
                        st.subheader("📊 Conversion Summary")
                        
                        # Group by emotion and usage
                        summary = df.groupby(['emotion', 'Usage']).size().reset_index(name='count')
                        
                        emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                        final_summary = []
                        
                        for i, emotion_name in enumerate(emotion_names):
                            train_count = summary[(summary['emotion'] == i) & (summary['Usage'] == 'Training')]['count'].sum()
                            test_count = summary[(summary['emotion'] == i) & (summary['Usage'] == 'PublicTest')]['count'].sum()
                            
                            final_summary.append({
                                'Emotion': emotion_name,
                                'Training': int(train_count),
                                'Testing': int(test_count),
                                'Total': int(train_count + test_count)
                            })
                        
                        summary_df = pd.DataFrame(final_summary)
                        st.dataframe(summary_df)
                        
                        # Download button
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV File",
                            data=csv_data,
                            file_name="fer2013_converted.csv",
                            mime="text/csv"
                        )
                        
                        st.markdown("""
                        ---
                        ## ✅ Success! Next Steps:
                        
                        1. **Use the generated CSV** in the main emotion detection app
                        2. **Go to "Model Training"** tab in the emotion detection system
                        3. **Upload the fer2013_converted.csv** file
                        4. **Start training** your emotion detection model!
                        
                        The CSV file is now in the same folder as your dataset.
                        """)
                        
                    else:
                        st.error("❌ No images were converted. Please check the folder structure.")
        
        else:
            st.error("❌ Could not find 'train' and 'test' folders. Please check the path.")
            st.write("**Current folder contents:**")
            try:
                contents = os.listdir(dataset_path)
                for item in contents:
                    st.write(f"- {item}")
            except:
                st.write("Could not list folder contents")
    
    elif dataset_path:
        st.error(f"❌ Folder not found: {dataset_path}")
        st.write("**Tips:**")
        st.write("- Make sure you extracted the ZIP file")
        st.write("- Use the full path to the folder")
        st.write("- Check for spaces or special characters in the path")

if __name__ == "__main__":
    main()