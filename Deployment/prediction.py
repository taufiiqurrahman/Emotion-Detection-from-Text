import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load Model
model = tf.keras.models.load_model('best_model')

# Function to get sentiment label
def get_sentiment_label(sentiment):
    return {
        0: 'Other',
        1: 'Sadness',
        2: 'Neutral',
        3: 'Worry',
        4: 'Love',
        5: 'Happiness'
    }.get(sentiment, 'Unknown Sentiment')

# Function to display sentiment icons
def display_sentiment_icon(sentiment):
    sentiment_lower = sentiment.lower()  # Convert to lowercase
    if sentiment_lower == 'other':
        st.image('https://img.freepik.com/premium-vector/set-handdrawn-emotional-character-faces-showing-different-emotions-feelings_511716-194.jpg', width=300)
    elif sentiment_lower == 'sadness':
        st.image('https://c1.wallpaperflare.com/preview/496/985/84/emotional-sad-childhood-boy.jpg', width=200)
    elif sentiment_lower == 'neutral':
        st.image('https://img.freepik.com/free-vector/smiling-face-expression_52683-32028.jpg', width=200)  # Corrected URL for neutral sentiment
    elif sentiment_lower == 'worry':
        st.image('https://c4.wallpaperflare.com/wallpaper/439/821/233/the-wolf-of-wall-street-wallpaper-preview.jpg', width=200)
    elif sentiment_lower == 'love':
        st.image('https://w0.peakpx.com/wallpaper/384/207/HD-wallpaper-love-fingers-stickers-emotions-emoji-red-lovely-contact-love-illusion-thumbnail.jpg', width=200)
    elif sentiment_lower == 'happiness':
        st.image('https://i.pinimg.com/originals/2e/94/d5/2e94d5765effa6c972be786fe567e3b8.jpg', width=200)



# Function to run the Streamlit app
def run():
    st.title('Sentiment Analysis App')
    st.write('Enter your comment and click the button to get sentiment analysis results.')

    y_pred_inf = None  # Initialize y_pred_inf with a default value

    # Create forms
    with st.form(key='sentiment'):
        text = st.text_input('Input your comment here', value='')
        submitted = st.form_submit_button('Analyze Sentiment')

    if submitted:
        try:
            # Predict new data inference using the model
            data_inf = pd.DataFrame({'text': [text]})
            y_pred_inf = np.argmax(model.predict(data_inf))

            # Display Result in Streamlit
            sentiment_label = get_sentiment_label(y_pred_inf)
            st.write(f'The comment is categorized as "{sentiment_label}" sentiment.')

            # Display sentiment icon
            display_sentiment_icon(sentiment_label)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    run()
