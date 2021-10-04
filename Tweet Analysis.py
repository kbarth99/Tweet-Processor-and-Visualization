# Import All Necessary Packages
import pandas as pd
import numpy as np
import preprocessor as prepro
import re
import tensorflow_hub as hub
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics.pairwise
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from sklearn.manifold import TSNE
import altair as alt  # Make Sure to Install altair_viewer
import spacy
nlp = spacy.load('en_core_web_sm')
import sys
from collections import Counter
import geopy 
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import FastMarkerCluster
import webbrowser
locator = geopy.geocoders.Nominatim(user_agent='mygeocoder')
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url) #download and load the model
print("Done!")


# Defining Stop Statements to Quit Functions
quit_statements = ['quit','end','stop']

# Creates Function to Interpret an Error Input
def iserror(func, args, itterlist = False):
   
    if np.logical_not(itterlist):
        try:
            func(args)
            return False
        except Exception:
            return True
    elif itterlist:
        try:
            [func(x) for x in args]
            return False
        except Exception:
            return True


# Filters Tweets with Like Count Greater than a Specific Value 
def helperMinLikes(data):
   
    minlikes = input("Minimum Number of Likes on Tweet? ")
    length = len(data)
    if np.logical_not(iserror(float,minlikes)):
        minlikes = float(minlikes)
        data = data.loc[data['likeCount']>=minlikes,:]
        print("Removed " + str(length-len(data))+ " Tweets")
        return data
    elif minlikes.lower() in quit_statements:
        sys.exit()
    else:
        print("Invalid Input, Digits Only")
        helperMinLikes(data)


# Filters Tweets with Like Count Greater than a Specific Value     
def helperMinFollowers(data):
   
    minfollowers = input("Minimum Number of Followers for Poster? ")
    length = len(data)
    if np.logical_not(iserror(float,minfollowers)):
        minfollowers = float(minfollowers)
        data = data.loc[data['followers']>=minfollowers,:]
        print("Removed " + str(length-len(data))+ " Tweets")
        return data
    elif minfollowers.lower() in quit_statements:
        sys.exit()
    else:
        print("Invalid Input, Digits Only")
        helperMinFollowers(data)


# Removes Specific Posters From Tweets
def helperRemovePosters(data):
    
    length = len(data)
    yesorno = input("Are there any other people/organizations you'd like to remove? [y/n] ")
    if yesorno.lower() in ['y','yes']:
        removetweeters = input("What people/organizations would you like to remove? (Separate with comma) ")
        toremove = removetweeters.split(',')
        toremove = [x.lower() for x in toremove]
        toremove = [x.strip() for x in toremove]
        data = data.loc[np.logical_not(data['username'].str.contains('|'.join(toremove))),:]
        print(f"Removed {length-len(data)} Tweets")
        return data
    elif yesorno.lower() in ['n','no']:
        return data
    elif yesorno.lower() in quit_statements:
        sys.exit()
    else:
        print("Invalid Input")
        helperRemovePosters(data)


# Removes Posts Containing a Specific Work or Phrase
def helperRemovePosts(data):
    
    length = len(data)
    yesorno = input("Are there any words you would not like to be included in your data?[y/n] ")
    if yesorno.lower() in ['y','yes']:
        removedata = input("What words should cause a tweet to not be included? (Separate with comma) ")
        if ',' in removedata:
            toremove = removedata.split(',')
        else:
            toremove = list(removedata)
        toremove = [x.lower() for x in toremove]
        toremove = [x.strip() for x in toremove]
        data = data.loc[np.logical_not(data['text'].str.contains('|'.join(toremove))),:]
        print(f"Removed {length-len(data)} Tweets")
        return data
    elif yesorno.lower() in ['n','no']:
        return data
    elif yesorno.lower() in quit_statements:
        sys.exit()
    else:
        print("Invalid Input")
        helperRemovePosts(data)


# Removes News Organizations Based on Content and Poster
def helperRemoveNews(data):
    
    news_outlets_names = 'mashable,cnnbrk,big_picture,theonion,time,breakingnews,bbcbreaking,espn,harvardbiz,gizmodo,techcrunch,wired,wsj,smashingmag,pitchforkmedia,rollingstone,whitehouse,cnn,tweetmeme,peoplemag,natgeosociety,nytimes,lifehacker,foxnews,waitwait,newsweek,huffingtonpost,newscientist,mental_floss,theeconomist,emarketer,engadget,cracked,slate,bbcclick,fastcompany,reuters,incmagazine,eonline,rww,gdgt,instyle,mckquarterly,enews,nprnews,usatoday,mtv,freakonomics,boingboing,billboarddotcom,empiremagazine,todayshow,good,gawker,msnbc_breaking,cbsnews,guardiantech,usweekly,life,sciam,pastemagazine,drudge_report,parisreview,latimes,telegraphnews,abc7,arstechnica,cnnmoney,nprpolitics,nytimesphoto,nybooks,nielsenwire,io9,sciencechannel,usabreakingnews,vanityfairmag,cw_network,bbcworld,abc,themoment,socialmedia2day,slashdot,washingtonpost,tpmmedia,msnbc,wnycradiolab,cnnlive,davos,planetmoney,cnetnews,politico,tvnewser,guardiannews,yahoonews,seedmag,tvguide,travlandleisure,newyorkpost,discovermag,sciencenewsorg'
    news_outlets_names = news_outlets_names.split(sep=',')
    news_outlets_names.extend(['news','stock','invest','update'])
    news_outlets_contents = ['article' , 'read more' , 'news' , 'new video','post','says.','said.','reports','breaking',"here's",'via ','"']
    yesorno = input("Remove News Outlets from Tweets? [y/n] ")
    if yesorno.lower() in ['y','yes']:
        length = len(data)
        data = data.loc[np.logical_not(data['username'].str.contains('|'.join(news_outlets_names))),:]
        data = data.loc[np.logical_not(data['text'].str.contains('|'.join(news_outlets_contents))),:]
        print(f"Removed {length-len(data)} Tweets")
        return data
    elif yesorno.lower() in ['n','no']:
        length = len(data)
        onlynews = input("Keep Only News Outlets? [y/n] ")
        if onlynews.lower() in ['y','yes']:
            data = data.loc[data['username'].str.contains('|'.join(news_outlets_names)),:]
            data = data.loc[np.logical_not(data['text'].str.contains('|'.join(news_outlets_contents))),:]
            print(f"Removed {length-len(data)} Tweets")
            return data
        elif onlynews.lower() in quit_statements:
            sys.exit()
        elif onlynews.lower() in ['n','no']:
            print("All Tweets Kept")
            return data
        else:
            print("Invalid Input")
            helperRemoveNews(data)
    elif yesorno.lower() in quit_statements:
        sys.exit()
    else:
        print("Invalid Input")
        helperRemoveNews(data)


# Removes Reply Tweets if Necessary
def helperRemoveReplies(data):
    
    yesorno = input("Remove Reply data?[y/n] ")
    if yesorno.lower() in ['y','yes']:
        length = len(data)
        data = data.loc[data.inReplyToTweetId.isna(),:]
        print(f"Removed {length-len(data)} Tweets")
        return data
    elif yesorno.lower() in ['n','no']:
        print("Replies Included")
        return data
    elif yesorno.lower() in quit_statements:
        sys.exit()
    else:
        print("Invalid Input")
        helperRemoveReplies(data)


# Removes Quoted (Retweeted) Tweets if Necessary
def helperRemoveQuoted(data):
    
    yesorno = input("Remove Quote data?[y/n] ")
    if yesorno.lower() in ['y','yes']:
        length = len(data)
        data = data.loc[data.quotedTweet.isna(),:]
        print(f"Removed {length-len(data)} Tweets")
        return data
    elif yesorno.lower() in ['n','no']:
        print("Quote Tweets Included")
    elif yesorno.lower() in quit_statements:
        sys.exit()
    else:
        print("Invalid Input")
        helperRemoveReplies(data)


def cleantweets(tweets):
    
    tweets['date'] = pd.to_datetime(tweets['date'])
    
    # Creates Follower Count Column
    tweets['followers'] = tweets.apply(lambda row: row['user']['followersCount'], axis = 1)
    liked = helperMinLikes(tweets)
    liked = helperMinFollowers(liked)
    
    # Creating Username and Display Name for Filtering
    liked['username'] = liked.apply(lambda row: row['user']['username'], axis = 1)
    liked['username'] = liked['username'].str.lower()
    liked['displayname'] = liked.apply(lambda row: row['user']['displayname'], axis = 1)
    liked['displayname'] = liked['displayname'].str.lower()
    filtered = helperRemovePosters(liked)
    filtered = helperRemoveReplies(filtered)
    filtered = helperRemoveQuoted(filtered)
    return filtered.filter(['id','date','content','username','likeCount','replyCount'], axis = 1)


# Uses Regex to Remove any Emojis
def helperRemoveEmojis(data):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    data = emoji_pattern.sub(r'', data)
    return data


def preprocess(tweets):
    
    # Applies Tweet Preprocessor
    prepro.set_options(prepro.OPT.URL, prepro.OPT.RESERVED, prepro.OPT.HASHTAG, prepro.OPT.EMOJI)
    tweets['text']  = tweets['content'].apply(lambda row: prepro.clean(row))
    tweets['text']= tweets['text'].apply(helperRemoveEmojis)
    
    # Fixes Any Issues that Preprocessor May Have Missed
    tweets['text'] =  [re.sub(r'&amp;', 'and', w) for w in tweets.text]
    htmlents = r'|'.join((r'&copy;',r'&reg;',r'&quot;',r'&gt;',r'&lt;',r'&nbsp;',r'&apos;',r'&cent;',r'&euro;',r'&pound;')) 
    tweets['text'] =  [re.sub(htmlents, '', w) for w in tweets.text]
    htag = r'|'.join((r"#", r"@"))
    tweets['text'] =  [re.sub(htag, '', w) for w in tweets.text]
    tweets['text'] = tweets.text.replace({' +':' '},regex=True)
    tweets['text'] = tweets.text.str.lower()
    tweets = helperRemovePosts(tweets)
    tweets = helperRemoveNews(tweets)
    
    # Removes Duplicate Tweets
    length = len(tweets)
    tweets = tweets.drop_duplicates(subset='text', keep="first")
    tweets = tweets.drop_duplicates(subset='id', keep="first")
    print("Removed " + str(length-len(tweets))+ " Duplicate Tweets")
    return tweets


def vectorize(tweets):
    
    # Creates Batches to Apply Vectorizer
    input_ids = tweets.index
    batch_size = 1000
    n_of_iterations = (input_ids.shape[0] + batch_size - 1) // batch_size
    FeatureVectors = pd.DataFrame(columns = ['TweetVectors'])
    
    # Creates Vectors for Tweets and Concatonates to Original DataFrame
    for i in range(n_of_iterations): 
        use_vecs = pd.DataFrame(index=tweets.loc[input_ids[i*batch_size:(i+1)*batch_size],['text']].index)
        embeddings = embed(tweets.loc[input_ids[i*batch_size:(i+1)*batch_size],['text']]['text'])
        use_vecs['TweetVectors'] = pd.Series(np.array(embeddings).tolist(), index=use_vecs.index) 
        FeatureVectors = pd.concat([FeatureVectors,use_vecs])
    tweets = pd.concat([tweets, FeatureVectors], axis=1)
    return tweets


def removeoutliers(tweets):
    
    tweet_embeddings = tweets.TweetVectors.to_list()
    
    # Applying Isolation Forest to Vectors to Find Outliers
    outliers_fraction = 0.1
    isf = IsolationForest(contamination=outliers_fraction,random_state=42, n_jobs=-1)
    y_pred = isf.fit(tweet_embeddings).predict(tweet_embeddings)
    
    # Indicates Which Values are Outliers
    result = np.where(y_pred == -1)
    remove=result[0].tolist()
    
    # Removes Non-Outliers from Tweet Embeddings
    for i in sorted(remove, reverse=True):
        del tweet_embeddings[i]
    
    # Removes Outliers form Original DataFrame
    length = len(tweets)    
    tweets.drop(tweets.index[remove], inplace=True)
    print(f"Number of outlier tweets removed: {length - len(tweets)}") 
    return tweets


# Applies Specified Number of Topics to Tweets
def helperNumberOfTopics(data):
  
    k=input("How many topics would you like? ")
    if np.logical_not(iserror(int,k)):
        kmeans = KMeans(n_clusters=int(k), init='k-means++', max_iter=1000, n_init=10, random_state=0)
        pred_y = kmeans.fit_predict(data.TweetVectors.to_list())+1
        data['Topic'] = pred_y
        
        # Plots Number of Tweets per Topic
        sns.set(color_codes=True)
        sns.set(rc={'figure.figsize':(5,5)})
        ax = sns.histplot(data['Topic'],
                      bins=int(k),
                      kde=False,
                      color='skyblue')
        ax.set(xlabel='Tweets per Cluster', ylabel='Frequency')
        return data
    
    elif k.lower() in quit_statements:
        sys.exit()
        
    else:
        print("Invalid Input")
        helperNumberOfTopics(data)


def topiccreation(tweets):
    
    # Applies K Means to Each number of topics in 1:10
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=1000, n_init=10, random_state=0)
        kmeans.fit(tweets.TweetVectors.to_list())
        wcss.append(kmeans.inertia_)
    
    plt.clf()
    # Plots Elbow Chart
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    return helperNumberOfTopics(tweets)
    

# Filters Tweets Based on Specified Topics
def helperChooseTopics(data):
    
    relevant = input("What topics are most relevant to you? (Separate values with commas) ")
    relevant = relevant.split(',')
    relevant = [x.strip() for x in relevant]
    if np.logical_not(iserror(int,relevant,itterlist=True)):
        relevant = [int(x) for x in relevant]
        data = data.loc[data.Topic.isin(relevant),:]
        return data.reset_index().drop(columns = "index")
    elif relevant.lower() in quit_statements:
        sys.exit()
    else:
        print("Invalid input, please only list integers")
        helperChooseTopics(data)
        
        
def relevanttopics(tweets):
    
    tweets['Relevance']=np.nan
    for t in range(1,tweets.Topic.max()+1):
    
        # Creates Topic DataFrame with Tweets of Topic t
        topic = tweets[['Topic','TweetVectors']][tweets.Topic==t]
    
        # Creates a Cosine Similarity Matrix
        cos_sim = sklearn.metrics.pairwise.cosine_similarity(topic.TweetVectors.to_list(),topic.TweetVectors.to_list())
        cos_sim[np.isnan(cos_sim)] = 1

        # Weighed Degree Centrality is the Sum of Cosine Similarities to Other Vectors in the Topic Minus Self-Similarity
        topic['Relevance']=pd.Series((np.sum(cos_sim,axis=1)-1).tolist()).values.round(2)
    
        # Re-Scales centraluty from 0 to 1
        topic['Relevance']=topic['Relevance']/topic['Relevance'].max()*100 
    
        # Drops Unneeded Columns
        topic.drop(topic.columns[[0, 1]], axis = 1, inplace = True)
    
        # Updates Original DataFrame with Centrality for Tweets in Topic T
        tweets.update(topic)
    
    # Prints Top 10 Most Relevant Tweets in Each Topic
    for t in range(1,tweets.Topic.max()+1):
        s = tweets[tweets.Topic==t].nlargest(10, columns=['Relevance']).text
        print("Topic "+str(t))
        for index in s.index:
            print(s[index])
        print("")
    
    return helperChooseTopics(tweets)
    

def sentanalysis(tweets):
    
    # Instantiates Sentiment Analyzer
    sid_obj = SentimentIntensityAnalyzer()
    
    # Creates Columns for Each Score in Original DataFrame
    key_list = "pos neu neg compound overall_sentiment".split(" ")
    for i in key_list:
      tweets[i] = ""
    
    # Analyze the Tweets Text Row by Row in Original DataFrame and Fills in Sentiment, Polarity, and Overall Sentiment
    for index, row in tweets.iterrows():
        row_sentiment_dict = sid_obj.polarity_scores(row['text'])
        
        # Rates Overall Sentiment Based on Compound Score
        if row_sentiment_dict['compound'] >= 0.05 : 
            row_sentiment_dict['overall_sentiment']  = "Positive"
    
        elif row_sentiment_dict['compound'] <= - 0.05 : 
            row_sentiment_dict['overall_sentiment'] =  "Negative"
    
        else : 
            row_sentiment_dict['overall_sentiment'] =  "Neutral"
            
        # Appends Scores and Sentiments to Original DataFrame 
        for i in key_list:
            tweets.loc[index, i] = row_sentiment_dict[i]
            
    # Drops Unneeded Scores from Original DataFrame and Prints Number of Each Sentiment        
    tweets = tweets.drop(columns = ['pos','neu','neg'])
    print(tweets['overall_sentiment'].value_counts())
    return tweets
   

# Filters Tweets to Only One Sentiment
def posorneg(tweets):
    
    question = input("Would you like to investigate postive or negative tweets? ")
    if question.lower() in ['positive','pos','positive tweets']:
        tweets = tweets.loc[tweets.overall_sentiment == "Positive",:].sort_values('compound', ascending = False).reset_index()
        print(f'Returning only {len(tweets)} positive tweets, Creating new Topics')
    elif question.lower() in ['negative','neg','negative tweets']:
        tweets = tweets.loc[tweets.overall_sentiment == "Negative",:].sort_values('compound', ascending = True).reset_index()
        print(f'Returning only {len(tweets)} negative tweets, Creating new Topics')
    elif question.lower() in quit_statements:
        sys.exit()
    else:
        print("Invalid Input")
        posorneg(tweets)
    return topiccreation(tweets)
    

def visualizetweets(tweets):
    
    # Applies TSNE to Tweet Sample, and Creates DataFrame to Plot
    tweetsample = tweets.sample(1000)
    X_tsne = TSNE(n_components=2, verbose=1, perplexity=25, n_iter=1000, learning_rate=25, random_state=21).fit_transform(tweetsample.TweetVectors.to_list())
    tweetsample['tSNE_X']=pd.Series((X_tsne[:, 0]).tolist()).values
    tweetsample['tSNE_Y']=pd.Series((X_tsne[:, 1]).tolist()).values
    source = pd.DataFrame(
    {'x': tweetsample['tSNE_X'],
     'y': tweetsample['tSNE_Y'],
     'txt': tweetsample["text"],
     'Topic' : tweetsample['Topic'],
     'Relevance' : tweetsample['Relevance']
    })

    # Define Bubbles on Map, Create Response to Mouse Hover
    bubbles = alt.Chart(source).mark_circle().encode(x=alt.X('x:Q', axis=alt.Axis(title="not directly interpretable", grid=False, labels=False),scale=alt.Scale(domain=[min(source.x)-10, max(source.x)+20])),y=alt.Y('y:Q', axis=alt.Axis(title="not directly interpretable", grid=False, labels=False),scale=alt.Scale(domain=[min(source.y)-10, max(source.y)+10])),color = 'Topic:N',
    tooltip=[alt.Tooltip('txt', title='Tweet'),
             alt.Tooltip('Topic', title='Topic'),
             alt.Tooltip('Relevance', title='Relevance')
            ]
    )
    
    # Visualizes Tweet Sample in an Interactive Map (Cannot Continue Until Tweet Map is Closed)
    bubbles.encode(text='txt').interactive().properties(height=1000,width=1000,title="Tweet Clusters (Sample of 1000 Tweets") 
    bubbles.show()


# Displays Top 20 Most Frequent Negative Adjective Noun Pairs 
def negadjnounpairs(tweets):
    
    complaints = []
    for index, row in tweets.iterrows():
        doc = nlp(row['text'])
        most_neg_adjective_score = -0.15
        second_most_neg_adjective_score = -0.1
        first_addition =''
        second_addition=''
        for token in doc:
            if token.dep_ == 'amod':
                score = analyser.polarity_scores(token.text+' '+token.head.text)['compound']
                if score < most_neg_adjective_score:
                    first_addition= token.text +' '+token.head.text
                    most_neg_adjective_score = score
                elif score<second_most_neg_adjective_score:
                    second_addition=token.text + ' '+token.head.text
                    second_most_neg_adjective_score = score
        if first_addition != '':
            complaints.append(first_addition)
        if second_addition != '':
            complaints.append(second_addition)
    print(Counter(complaints).most_common(20))
    

# Displays Top 20 Most Frequent Positive Adjective Noun Pairs
def posadjnounpairs(tweets):
    
    praise = []
    for index, row in tweets.iterrows():
        doc = nlp(row['text'])
        most_pos_adjective_score = 0.15
        second_most_pos_adjective_score = 0.1
        first_addition =''
        second_addition=''
        for token in doc:
            if token.dep_ == 'amod':
                score = analyser.polarity_scores(token.text+' '+token.head.text)['compound']
                if score > most_pos_adjective_score:
                    first_addition= token.text +' '+token.head.text
                    most_pos_adjective_score = score
                elif score>second_most_pos_adjective_score:
                    second_addition=token.text + ' '+token.head.text
                    second_most_pos_adjective_score = score
        if first_addition != '':
            praise.append(first_addition)
        if second_addition != '':
            praise.append(second_addition)
    print(Counter(praise).most_common(20))


# Removes Specified Locations from Mentioned Locations in Tweets
def helperCleanGPE(gpelist):
    
    print(Counter(gpelist).most_common())
    yesorno = input("Are there any locations listed that you'd like to remove? [y/n] ")
    if yesorno.lower() in ['y','yes']:
        removelocations = input("What locations would you like to remove? (Separate with comma) ")
        toremove = removelocations.split(',')
        toremove = [x.lower() for x in toremove]
        toremove = [x.strip() for x in toremove]
        cleaned = []
        for i in gpelist:
            if np.logical_not(any(x in i for x in toremove)):
                cleaned.append(i)
        return cleaned
    elif yesorno.lower() in ['n','no']:
        return gpelist
    elif yesorno.lower() in quit_statements:
        sys.exit()
    else:
        print("Invalid Entry")
        helperCleanGPE(gpelist)


def mapofmentions(tweets):
    
    # Creates List of Mentioned Locations within Tweets
    gpe_dirty = []
    for index, row in tweets.iterrows():
        doc = nlp(row['text'])
        to_add = []
        for ent in doc.ents:
            if (ent.label_ == 'GPE') and (ent.text not in to_add):
                to_add.append(ent.text)
        gpe_dirty.extend(to_add)
    gpe_clean = helperCleanGPE(gpe_dirty)
    print('Cleaned, Applying Coordinates')
    
    # Splits Out Unique Values for Locations for Efficiency
    gpe_full = pd.DataFrame(gpe_clean).rename(columns = {0:'location'})
    gpe_df = pd.DataFrame(set(gpe_clean)).rename(columns = {0:'location'})
    
    # Applies Coordinates to Each Location
    gpe_df['address'] = gpe_df['location'].apply(geocode)
    gpe_df['coordinates']=gpe_df['address'].apply(lambda loc: tuple(loc.point) if loc else 'None')
    gpe_df = gpe_df[gpe_df['coordinates']!='None']
    print('Coordinates Applied')
    
    # Remerges Coordinates and All Mentions
    gpe_df = gpe_full.merge(gpe_df, on = 'location')
    
    # Splits out Latitude, Longitude and Altitude of Mentioned Locations
    gpe_df[['latitude','longitude','alititude']]=pd.DataFrame(gpe_df['coordinates'].tolist(), index = gpe_df.index)
    gpe_df = gpe_df.loc[pd.notnull(gpe_df['latitude'])]
    
    # Initiates Folium Map and Plots All Latitudes and Longitudes
    folium_map = folium.Map(location=[59.338315,18.089960],zoom_start = 2, tiles = 'cartoDB dark_matter')
    FastMarkerCluster(data = list(zip(gpe_df['latitude'].values,gpe_df['longitude'].values))).add_to(folium_map)
    folium.LayerControl().add_to(folium_map)
    
    # Saves Map as HTML and Opens in Web Browser
    folium_map.save("map.html")
    webbrowser.open("map.html")



# Load Scraped Tweet File Using SNScrape
loadtweets = pd.read_json("marriott.json", lines = True)

# Cleaning and Filtering Functions
cleaned = cleantweets(loadtweets)
preprocessed = preprocess(cleaned)
vectorized = vectorize(preprocessed)
outlierremoved = removeoutliers(vectorized)
topicadded = topiccreation(outlierremoved)
mostrelevant = relevanttopics(topicadded)
sented = sentanalysis(mostrelevant)
negative = posorneg(sented)

# Visualization Functions
visualizetweets(negative)
negadjnounpairs(negative)
mapofmentions(negative)





