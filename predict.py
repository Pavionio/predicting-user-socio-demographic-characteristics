class PredictSexAge():
    def __init__(self, test_events=test_events,train_events=train_events,
                 train_targets=train_targets,video_info_v2=video_info_v2,
                 tzvoc=tzvoc,tokenizer=tokenizer):
        from zoneinfo import ZoneInfo
        import pandas as pd
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import KBinsDiscretizer
        from sklearn.compose import ColumnTransformer
        self.test_events=test_events
        self.train_events=train_events
        self.train_targets=train_targets
        self.video_info_v2=video_info_v2
        self.tzvoc=tzvoc
        self.tokenizer=tokenizer
        self.tarvoc={i:dict(zip(train_targets['viewer_uid'],train_targets[i])) 
                     for i in ['sex', 'age_class']}
        self.vidvoc={i:dict(zip(video_info_v2['rutube_video_id'],video_info_v2[i])) 
                     for i in ['title', 'category', 'duration', 'author_id']}
        print('vocs...done')
        self.test_events['event_timestamp']=pd.to_datetime(self.test_events['event_timestamp'])    
        self.train_events['event_timestamp']=pd.to_datetime(self.train_events['event_timestamp']) 
        def time2local(x):
            return x['event_timestamp'].astimezone(ZoneInfo(self.tzvoc[x['region']]))
        self.test_events['local_event_timestamp']=pd.to_datetime(self.test_events[['event_timestamp','region']
        ].progress_apply(time2local, axis=1), utc=True)                                          
        self.train_events['local_event_timestamp']=pd.to_datetime(self.train_events[['event_timestamp','region']
        ].progress_apply(time2local, axis=1), utc=True)
        self.test_events['local_hour']=self.test_events['local_event_timestamp'].dt.hour
        self.test_events['local_dayofweek']=self.test_events['local_event_timestamp'].dt.dayofweek
        self.train_events['local_hour']=self.train_events['local_event_timestamp'].dt.hour
        self.train_events['local_dayofweek']=self.train_events['local_event_timestamp'].dt.dayofweek
        print('time...done')
        for i in ['title', 'category', 'duration', 'author_id']:
            self.test_events[i]=self.test_events['rutube_video_id'].map(self.vidvoc[i])
            self.train_events[i]=self.train_events['rutube_video_id'].map(self.vidvoc[i])
        self.test=pd.DataFrame()
        self.train=pd.DataFrame()
        for i in tqdm(['region', 'ua_device_type', 'ua_client_type', 
                       'ua_os', 'ua_client_name', 'rutube_video_id',
                       'title', 'category', 'author_id',
                       'local_hour','local_dayofweek']):
            self.train[i]=train_events[i].astype(str).fillna('nan').groupby(
                self.train_events['viewer_uid']).agg(lambda x: ' '.join(list(x)))
            self.test[i]=train_events[i].astype(str).fillna('nan').groupby(
                self.test_events['viewer_uid']).agg(lambda x: ' '.join(list(x)))
        for i in tqdm(['total_watchtime','duration']):
            self.train[i]=self.train_events[i].groupby(self.train_events['viewer_uid']
                        ).median()  
            self.test[i]=self.test_events[i].groupby(self.test_events['viewer_uid']
                        ).median() 
        self.train['viewer_uid']=self.train.index.tolist()
        self.test['viewer_uid']=self.test.index.tolist()
        for i in tqdm(['sex', 'age_class']):
            self.train[i]=self.train['viewer_uid'].map(self.tarvoc[i])
        print('train,test...done')
        tw=dict(zip(['region', 'ua_device_type', 'ua_client_type', 'ua_os', 'ua_client_name',
       'total_watchtime', 'category', 'duration',
       'local_hour', 'local_dayofweek', 'title', 'rutube_video_id',  'author_id'],[.1]*10+[1.]*3))
        self.s_pipe=Pipeline([('input',ColumnTransformer([
                ('scaler',KBinsDiscretizer(50,strategy='kmeans'),['total_watchtime','duration']),
                ]+[(i, TfidfVectorizer(tokenizer=self.tokenizer.tokenize, token_pattern=None),i) for i in ['title']
                ]+[(i, TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b'),i) for i in ['region', 'ua_device_type', 
                'ua_client_type', 'ua_os', 'ua_client_name', 'category',
                'local_hour', 'rutube_video_id',  'author_id']], transformer_weights=tw)),
               ('rgs',LinearSVC(C=0.01,class_weight='balanced'))
                ]).fit(self.train.fillna('nan'), 
                       self.train['sex'])
        self.a_pipe=Pipeline([('input',ColumnTransformer([
                ('scaler',KBinsDiscretizer(50,strategy='kmeans'),['total_watchtime','duration']),
                ]+[(i, TfidfVectorizer(tokenizer=self.tokenizer.tokenize, token_pattern=None),i) for i in ['title']
                ]+[(i, TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b'),i) for i in ['region', 'ua_device_type', 
                'ua_client_type', 'ua_os', 'ua_client_name', 'category',
                'local_hour', 'local_dayofweek', 'rutube_video_id',  'author_id']], transformer_weights=tw)),
               ('rgs',LinearSVC(C=.1,class_weight='balanced'))
                ]).fit(self.train.fillna('nan'), 
                       self.train['age_class'])
        print('pipes.fit...done')
    def predict(self,X):
        test=self.test.loc[X['viewer_uid'].tolist()]
        X['sex']=self.s_pipe.predict(test)
        X['age_class']=self.a_pipe.predict(test)
        return X
