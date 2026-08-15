## Problem Statement

- I have a corpus or album of photos and videos. From this corpus I want to generate a moments video containing the most relevant clips and photos from the corpus
- I need this application to run on local laptop which is Mac m5 pro processor
- The corpus can be of a single trip or collection of multiple trips or photos taken in last N months or years
- User will input the location or path of the folder which has all the corpus either in that single directory or sub-dirs. User will also give an input like "Moments from the trek with epic mountain views" or "Make a video of my trip with friends that has most bonding and beautiful moments" or "Give me a video from the collection of beach trips". User will give a video length in seconds like 120s, 240s etc that will tell us the length video


- Future scope : We need to enable a feature where user can add photos of people and tag them with name. App has to find relevant photos with these people and generate video with them


## Constraints

- The corpus can have max 20 GB worth of data. We can take this constraint
- Output video length can never be more than 300s i.e. 5 minutes
- This has to run locally on m5 or m5 pro laptop
- Not putting any constriants of latency. Lets build the basic appliation first and then move to optimisations
- We are not generating anything here i.e. not using generative models to generate video. We are taking clips from user and stiching them with transitions. No artificial moments just captured moments stiched best

## Possibilities

- The corpus can be from a trek. This will have both people, mountains, river crossings, epic views etc. From the user's input string we need to find the most relevant and epic moments
- THis can be a corpus of fun trip with friends that can have fun moments or partying moments or just random dancing moments. Based on user's query we need to give them the best 
- This can be 5 years worth of data and user says to find the best bonding moments with friends and give me the output

and many such similar things.