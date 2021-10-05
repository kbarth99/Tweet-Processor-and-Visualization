# Tweet Processor and Visualization Tool

This project takes tweets scaped using SNScrape and asks for user input for preferences in filtering and cleaning tweets. The tweets are vectorized and clustered with a user-indicated number of topics using Tensorflow's Universal Sentence Encoder and K-Means. Vader sentiment analysis groups the tweets by positive and negative sentiment and the chosen sentiment's topic clusters are visualized using an interactive map from Altair. NLP is applied to the tweets to discover most frequently mentioned adjective-noun pairs in order to show most common complaints and praises using Spacy. Finally, tweets of the selected sentiment are scanned for locations mentioned using Spacy, which are then converted to GPS coordinates using GeoPy and plotted on a world map using Folium.

This tweet processor was first a project in BUSI/COMP 488: "Data Science in the Business World" at UNC-Chapel Hill. I have completely reworked the project and added multiple new facets in order to make the process generalizable to any tweets scraped with SNScrape. 

The dataset provided in this repository is a 738MB file of tweets including the word "Marriott" from 3/11/2020 through 3/11/2021. This program should be run on GPU if you'd like to include all tweets instead of those with a minimum like count of one, as the dataset will have long load times on CPU if all tweets are analyzed. 

Packages Required:
* Tensorflow/Tensorflow Hub
* Sci-kit Learn
* Vader Sentiment
* Altair (+ Altair Viewer if ran in Spyder)
* Spacy (+ en_core_web_sm)
* GeoPy
* Folium
* WebBrowser



*Note this project was not developed for profit-seeking use and is intended as an intuitive way of exploring topics on Twitter. Created for educational purposes only.*
