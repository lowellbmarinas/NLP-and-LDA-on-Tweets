#Applying NLP, Latent Dirichlet Allocation clustering, and Sentiment Analysis to real-time Twitter data via an API
#Scripted by Lowell B Marinas lowell.marinas@stern.nyu.edu/Follow me on Twitter, Instagram, GitHub @lowellbmarinas
#Published to GitHub on: 11 March 2015

#ENTER YOUR TWITTER CREDENTIALS
library(twitteR) #Refer to the following webpage: http://thinktostart.com/twitter-authentification-with-r/
api_key <- "PASTE YOUR TWITTER API KEY HERE BETWEEN THE QUOTATION MARKS"
api_secret <- "PASTE YOUR TWITTER API SECRET HERE BETWEEN THE QUOTATION MARKS"
access_token <- "PASTE YOUR TWITTER ACCESS TOKEN KEY HERE BETWEEN THE QUOTATION MARKS"
access_token_secret <- "PASTE YOUR TWITTER ACCESS TOKEN SECRET HERE BETWEEN THE QUOTATION MARKS"
#This is where you have to manually select
setup_twitter_oauth(api_key,api_secret,access_token,access_token_secret) #RUN CODE UNTIL HERE

#AFTER MANUAL SELECTION, YOU CAN PROCEED TO RUN THE REST OF THIS CODE
library(tm)
library(RTextTools)
library(topicmodels)
library(RWeka)
library(igraph)

#RETRIEVE TWITTER DATA
twitter_feed <- searchTwitter('nyustern', n=500) #Replace 'nyustern' with your own term to search. Change the number to determine how many tweets you would like
df <- do.call("rbind", lapply(twitter_feed, as.data.frame))

#CORPUS CREATION - DTM PREP
myCorpus <- Corpus(VectorSource(df$text))
myCorpus <- tm_map(myCorpus,content_transformer(function(x) iconv(x, to='UTF-8-MAC', sub='byte')),mc.cores=1)
myCorpus <- tm_map(myCorpus, content_transformer(tolower), mc.cores=1)
myCorpus <- tm_map(myCorpus, stripWhitespace, mc.cores=1)
myCorpus <- tm_map(myCorpus, removePunctuation, mc.cores=1)
myCorpus <- tm_map(myCorpus, removeNumbers, mc.cores=1)
myCorpus <- tm_map(myCorpus, function(x)removeWords(x,stopwords()),mc.cores=1)
myCorpus <- tm_map(myCorpus, removeWords, c("rt")) 

#DTM WITH N-GRAM CREATION
require(RWeka)
options(mc.cores=1)
TrigramTokenizer <- function(x) {RWeka::NGramTokenizer(x, RWeka::Weka_control(min = 1, max = 3))} #establishes NGrams
myDtm = DocumentTermMatrix(myCorpus,control = list(tokenize = TrigramTokenizer))

#WORDCLOUD GENERATION
library(wordcloud)
m = as.matrix(myDtm)
v = sort(colSums(m), decreasing=TRUE)
myNames = names(v)
k = which(names(v)=="miners")
myNames[k] = "mining"
d = data.frame(word=myNames, freq=v)
wordcloud(d$word, colors=c(1,2,4),random.color=TRUE,d$freq, min.freq=555) #decrease/increase min.freq to increase/decrease words shown 

#LATENT DIRICHLET ALLOCATION
k = 25 #Number of clusters
SEED = 12345 # set seed to allocate resources for the Mixture Model
my_TM =
  list(VEM = LDA(myDtm, k = k, control = list(seed = SEED)),
       VEM_fixed = LDA(myDtm, k = k,
                       control = list(estimate.alpha = FALSE, seed = SEED)),
       Gibbs = LDA(myDtm, k = k, method = "Gibbs",
                   control = list(seed = SEED, burnin = 1000,
                                  thin = 100, iter = 1000)),
       CTM = CTM(myDtm, k = k,
                 control = list(seed = SEED,
                                var = list(tol = 10^-4), em = list(tol = 10^-3))))
Terms = terms(my_TM[["VEM"]], 5) #number will spit out how many top terms for each topic in LDA
Terms #This will label each topic/cluster by the top terms indicated by line above
write.csv(Terms, file="Topic_Label.csv", row.names=FALSE)
topic<-(my_topics = topics(my_TM[["VEM"]])) #spits out the cluster that each doc belongs to
most_frequent = which.max(tabulate(my_topics)) #Discover most frequent words in the topics
frequent <- terms(my_TM[["VEM"]], 15)[, most_frequent] #Displays top 10 words in the topics.  You can use these most frequent terms to visualize the network of words
frequent <- as.list(frequent) #To be used for the Network of Words

#VISUALIZE THE NETWORK OF WORDS
require(RWeka)
options(mc.cores=1)
myTdm = TermDocumentMatrix(myCorpus,control = list(tokenize = TrigramTokenizer,dictionary=frequent))
myTdm <- as.matrix(myTdm)
myTdm[myTdm>=1] <- 1 # change it to a Boolean matrix
termMatrix <- myTdm %*% t(myTdm) # transform into a term-term adjacency matrix
g <- graph.adjacency(termMatrix, weighted=T, mode = "undirected") # build a graph from the above matrix
g <- simplify(g) # remove loops
V(g)$label <- V(g)$name # set labels and degrees of vertices
V(g)$degree <- degree(g)
set.seed(3952) # set seed to make the layout reproducible
layout1 <- layout.fruchterman.reingold(g)
V(g)$label.color <- rgb(0, 0, .2, .8)
V(g)$frame.color <- NA
egam <- (log(E(g)$weight)+.4) / max(log(E(g)$weight)+.4)
E(g)$color <- rgb(.5, .5, 0, egam)
E(g)$width <- egam
plot(g, layout=layout1) # plot the graph in layout1
plot(g, layout=layout.kamada.kawai)

#POSITIV COUNT
positiv_dtm<-as.data.frame(inspect(DocumentTermMatrix(myCorpus,list(dictionary=c("accessible","acclaim","acclamation","accolade","accompaniment","acquaintance","adherent","adhesion","adhesive","advocacy","affability","affable","affiliation","affirmative","allegiance","allies","allure","altruistic","amenity","amiability","amiable","amicable","amour","amusement","angel","angelic","art","availability","backing","ball","balmy","baptism","beacon","behalf","benefactor","beneficiary","benefit","benevolent","betrothal","bliss","blithe","bonus","breadwinner","brotherly","carefree","caress","ceremonial","charitable","charity","charm","chaste","cheerful","cheery","cherub","chivalrous","chivalry","chum","civilize","classic","cleanliness","clear","closeness","comedy","comestible","comic","comical","commemoration","commendation","common","communal","commune","communicate","communicative","companion","companionship","confederation","confidant","congenial","congratulatory","conscientious","considerate","constancy","consult","consultation","contentment","contributor","cordial","correction","courteous","courtesy","courtly","covenant","crusade","crusader","culture","cupid","dance","dear","decoration","delight","dig","discreet","discretion","discuss","donation","eagerness","earnestness","ease","ecstasy","ecstatic","education","educational","elate","elegance","empathy","enchant","enchantment","ensemble","entertainment","enthusiasm","excitedness","exertion","exhilaration","fairness","faithfulness","fellow","festive","festivity","filial","fit","flawless","fluent","fond","fondness","forward","friend","fun","gaiety","gaily","gallantry","generosity","giddy","gift","gladden","gladness","gleam","glee","glow","gold","golden","grace","handy","happiness","happy","haven","healthful","home","honeymoon","human","humanitarian","humorous","hygiene","inauguration","intelligible","intimacy","joke","jolly","joy","joyful","jubilant","jubilee","kid","learn","liberal","lifelong","light","loveliness","lover","luxury","lyric","lyrical","magnetic","majesty","make","marital","marriage","marry","mate","melody","merciful","merrily","merriment","merry","mint","miracle","miraculous","mirth","mobility","modernity","mutual","niche","nourishment","nutrient","oasis","objective","onward","open","optional","outgoing","overjoyed","palatable","palatial","paradise","patriot","patriotic","persuasive","playful","playmate","plaything","pleasantry","pleasurable","poetic","pomp","popularity","populous","posterity","practicable","practical","precept","prettily","pretty","productivity","propitious","prosperity","purification","purity","rapport","receptive","reconcile","refinement","refuge","reinforcement","relaxation","renovation","respite","restful","reunion","romance","romantic","rosy","safety","salutation","sanctuary","sane","sanitary","savings","security","shelter","skill","sociable","splendor","spotless","staple","stupendous","stylish","subscription","subsidy","subsistence","summit","super","supportive","sweet","sweetheart","synthesis","tact","tenderness","therapeutic","thoughtful","togetherness","train","tranquility","treasure","trust","trustworthiness","upfront","upside","upward","usefulness","utilitarian","utilization","valiant","viable","warmhearted","welfare","well","whimsical","wholesome","willful","wise","workable","worth","worthwhile","zenith")))))
positiv_dtm<-rowSums(positiv_dtm)

#NEGATIV COUNT
negativ_dtm1<-as.data.frame(inspect(DocumentTermMatrix(myCorpus,list(dictionary=c("abyss","adulterate","alien","anomalous","belated","bloody","congested","congestion","contagious","contradiction","controversy","coolness","cost","costliness","darken","daunting","deviation","dirt","disclaim","disguise","dungeon","evasion","exit","expense","farce","feudal","fleeting","frigid","grave","hag","haphazard","heedless","horde","hunger","hungry","immobility","impasse","intruder","jeopardy","marginal","melodramatic","miser","motionless","nosey","objection","obstruction","outbreak","outburst","outcry","outsider","partition","poison","propaganda","pry","qualm","rigor","rubbish","ruin","rumor","scream","screech","secede","secession","secrecy","secret","sedentary","shark","shipwreck","shroud","skulk","slime","snare","spinster","stagnant","stalemate","standstill","stray","substitution","tariff","temper","throw","topple","underworld","uproar","utterance","vomit","absurdity","accusation","admonition","adulteration","adultery","adversity","affectation","against","agitator","ailment","alienate","altercation","anarchist","anomaly","antagonist","antiquated","antitrust","artificial","assailant","assassin","avarice","avaricious","babble","backward","bait","banal","bandit","banishment","beggar","bereave","bereft","bizarre","blah","bland","blind","boisterous","bombardment","boredom","brat","bribe","brittle","broke","bruise","bug","bum","burden","burn","bury","busybody","callous","cancel","cancellation","cancer","capricious","capsize","careless","carelessness","casualty","chafe","charge","cheater","choke","chore","clamorous","clatter","clique","close","clumsy","clutter","coarse","coercive","cold","collision","collusion","combatant","commit","complicate","complication","conceal","concern","confinement","conspirator","conspire","contradict","contrary","controversial","cool","corrosion","corrosive","costly","covert","cranky","craze","craziness","critic","crook","culpable","curtail","dark","darkness","darn","dawdle","daze","deadweight","deaf","deafness","debatable","debtor","defame","defendant","dent","depress","derisive","desperation","deviate","devilish","dim","din","disease","dishonest","disingenuous","disinterest","distort","distortion","distract","distracting","dizzy","doomsday","dope","downcast","downhearted","drab","dreary","drowsiness","drowsy","dump","dunce","dwindle","embarrass","emergency","encroach","enrage","enslave","entangle","entanglement","epidemic","erosion","error","estranged","exasperate","exasperation","exhaust","exhaustion","expedient","expensive","fabricate","fabrication","fallout","famished","fascist","fat","fatal","fearsome","feign","feint","ferocious","ferocity","feud","fiend","fierce","filth","filthy","flimsy","fool","fragile","freak","frighten","front","fugitive","gash","germ","ghetto","glum","grievance","growl","guise","haggard","hang","harassment","hardship","hassle","hazy","headache","heartless","heinous","hoard","humiliation","hunt","hustler","hypocrisy","hypocrite","idleness","ill","immovable","impersonal","imprecision","inconsistency","incurable","indifferent","ineffectiveness","ineffectualness","inefficiency","infection","infest","inflation","infraction","ingratitude","insolence","insolent","intrude","intrusion","inundated","involuntary","involve","isolate","jail","jeopardize","jerk","jittery","jobless","jumpy","lamentable","lapse","leakage","liable","lifeless","limitation","limp","litter","lonesome","low","lull","lure","lurk","madman","malady","manipulation","meddle","mediocre","melancholy","mischief","mishap","mockery","moody","motley","mourner","mumble","murderous","nasty","naughty","neurotic","noise","nonchalant","nonsense","novice","nuisance","numb","nuts","obsolete","obstinate","opinionated","orphan","outcast","overbearing","owe","pain","painful","paltry","paralysis","paralyzed","parasite","passe","pathetic")))))
negativ_dtm2<-as.data.frame(inspect(DocumentTermMatrix(myCorpus,list(dictionary=c("perish","persecution","pervert","pessimism","pessimistic","pest","petty","pinch","pitiful","plaintiff","poisonous","pollution","pompous","pout","precipitate","presumptuous","procrastination","prod","quarrelsome","race","ramble","rat","recession","recklessness","refrain","refugee","regardless","remorse","renunciation","resignation","restless","restlessness","revenge","revoke","rigid","rigorous","rip","risky","rivalry","rogue","rotten","ruffian","runaway","rupture","rusty","ruthlessness","sarcasm","sarcastic","savage","scary","scoff","scornful","scream","scuffle","scum","selfishness","serve","shabby","shaggy","shallow","shameful","shortcoming","shrew","shrill","shrivel","shrug","shyness","simplistic","sinful","skirmish","slanderous","sleepless","sloppy","slump","smuggle","sneak","sneer","snore","somber","sore","soreness","sorrowful","sour","spiteful","spot","sputter","squander","stain","stammer","static","straggler","strain","stress","stuffy","stunt","stupid","stupidity","sullen","susceptible","symptom","tantrum","tardy","tempest","temporarily","terror","terrorism","thirst","thirsty","thorny","thud","timidity","tiresome","toil","tolerable","torrent","torturous","transgress","trap","traumatic","treachery","treasonous","trespass","turmoil","twitch","ultimatum","unaccustomed","unarm","unbearable","uncivil","unclean","uncouth","unfeeling","unguarded","unhappiness","unhealthy","uninformed","unmoved","unnatural","unprofitable","unruly","unsafe","unsettling","untimely","untruth","upheaval","uprising","uproot","vagabond","vagrant","vain","vanity","vehement","venomous","villain","viper","volatile","volatility","warlike","warp","waste","wayward","weed","wench","wickedness","woe","woeful","worn","worry","wrath","yawn")))))
negativ_dtm<-cbind(negativ_dtm1,negativ_dtm2)
negativ_dtm<-as.data.frame(negativ_dtm)
negativ_dtm<-rowSums(negativ_dtm)
rm(negativ_dtm1)
rm(negativ_dtm2)

#CALCULATES POSITIV VS NEGATIV POLARITY SCORE
polarity <- (positiv_dtm-negativ_dtm)/(positiv_dtm+negativ_dtm)
polarity <- as.data.frame(polarity)
polarity <- rapply(polarity, f=function(x) ifelse(is.nan(x),0,x), how="replace" ) #Replaces NaN's with 0's

#COMBINES TEXT WITH ITS TOPIC AND POLARITY SCORE
final_results <- cbind(topic,polarity,df)
View(final_results)
write.csv(final_results,file="Topic_Pol_score_on_Twitter_data.csv", row.names=FALSE)