import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
# from wordcloud import WordCloud, STOPWORDS


class AnalysisData():
    def __init__(self) -> None:
        pass

    def _run_analysis_(self, json_path, saved_dir):
        # create new saved folder
        if os.path.exists(saved_dir) is False:
            os.mkdir(saved_dir)
        # loading json
        entity_df, relation_df, len_of_tokens, document_list = self.load_json(json_path)
        # print(entity_df)
        entity_types = entity_df['label'].unique()
        relation_types = relation_df['label'].unique()
        self.get_data_types(entity_types, relation_types, saved_dir=os.path.join(saved_dir, "data_types.json"))

        # visualize sequence length
        self.visualize_histogram(len_of_tokens,x_label="num_tokens",y_label="frequency", 
                            saved_path=saved_dir+"/token_in_document_histogram.png", n_bins=None, range=None)
        # visualize entity and relation statitiscal
        self.visualize_type_bar(entity_df, saved_dir+"/entity_type.png", x_label="Entity types", y_label="Count of sample")
        self.visualize_type_bar(relation_df, saved_dir+"/relation_type.png", x_label="Relation types", y_label="Count of sample")

        # visualize co-apperance relations
        # print(relation_df)
        self.visualize_relation(relation_df, saved_name=saved_dir+"/correlation_entity_pairs", entity_types=entity_types)


        

    def load_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(open(json_path)) 

        document_list =[]
        entity_df = pd.DataFrame(columns=['doc_index','entity','label'])
        relation_df = pd.DataFrame(columns=['doc_index','entity_1','entity_2','entity_type_1','entity_type_2','label'])
        len_of_tokens = []
        for i, sample in enumerate(data):
            # print(sample.keys())
            tokens = sample["tokens"]
            entities = sample["entities"]
            relations = sample["relations"]
            doc_index = sample['orig_id']


            entity_doc_df = self.enumerate_entitiy(entities, tokens, doc_index)
            relation_doc_df = self.enumerate_relations(relations, entity_doc_df, tokens, doc_index)

            entity_df = pd.concat((entity_df, entity_doc_df))
            relation_df = pd.concat((relation_df, relation_doc_df))
            len_of_tokens.append(len(tokens))
            document_list.append(" ".join(tokens))

        print("Number of sample:", len(data))
        return entity_df, relation_df, len_of_tokens, document_list

    def enumerate_relations(self, relations, entities, tokens, doc_index):
        df = pd.DataFrame(columns=['doc_index','entity_1','entity_2','entity_type_1','entity_type_2','label'])

        for idx, rel in enumerate(relations):
            rel_label = rel['type']
            entity_1 = entities.iloc[int(rel['head'])]
            entity_2 = entities.iloc[int(rel['tail'])]

            df.loc[idx] = [doc_index, entity_1['entity'], entity_2['entity'], entity_1['label'], entity_2['label'], rel_label]

        return df

    def enumerate_entitiy(self, entities, tokens, doc_index):
        sub_df = pd.DataFrame(columns=['doc_index','entity','label'])
        for idx, entity in enumerate(entities):
            type = entity['type']
            start = entity['start']
            end = entity['end']

            entity_text = " ".join(tokens[start:end])

            sub_df.loc[idx] = [doc_index, entity_text, type]

        # print(sub_df)
        return sub_df
    
    def build_word_clound(self, seqs, saved_path):
        all_text = " ".join(seqs)
        word_cloud = WordCloud(width=3000, height=2000,
                                collocations=False, stopwords=STOPWORDS).generate(all_text)
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.savefig(saved_path)
        plt.close('all')

    def get_data_types(self, entity_types, relation_types, saved_dir):
        ent_dic = {}
        for ent in entity_types:
            ent_dic[ent] = dict(short=ent, verbose=ent)

        rel_dic = {}
        for rel in relation_types:
            rel_dic[rel] = dict(short=rel, verbose=rel, symmetric=False)


        data_types = dict(entities=ent_dic, relations=rel_dic)
        with open(saved_dir, "w") as file:
            json.dump(data_types, file, indent=2)

        



    def visualize_type_bar(self, dataframe, saved_name, x_label, y_label, color="cornflowerblue"):
        plt.figure()
        sns.set(font_scale=1.4)
        ax = dataframe['label'].value_counts().plot(kind='bar', figsize=(15, 6), rot=0, color=color)
        plt.xlabel(x_label, labelpad=14)
        plt.ylabel(y_label, labelpad=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for container in ax.containers:
            ax.bar_label(container)
        plt.savefig(saved_name, bbox_inches='tight')
        plt.close('all')

    def visualize_histogram(self, data, x_label, y_label, saved_path, n_bins, range, title=None):
        plt.figure(figsize=(10, 6))
        plt.hist(data, lw=1, ec="yellow", fc="red", alpha=0.5, bins=n_bins, range=range)
        plt.xlabel(xlabel=x_label)
        plt.ylabel(ylabel=y_label)
        plt.title(title)
        plt.savefig(saved_path, bbox_inches='tight')
        plt.close()

    def co_appearance(self, col1, col2):
        n11 = 0
        n10 = 0
        n01 = 0
        n00 = 0
        for i in range(len(col1)):
            if col1[i]==col2[i]==1:
                n11 += 1
            elif col1[i]==1 and col2[i]==0:
                n10 += 1
            elif col1[i]==0 and col2[i]==1:
                n01 += 1
            elif col1[i]==0 and col2[i]==0:
                n00 += 1

        return n11, n10, n01, n00
        # break
    def visualize_relation(self, relation_df, saved_name, entity_types):
        if os.path.exists(saved_name) is False:
            os.mkdir(saved_name)
        rel_types = relation_df['label'].unique()
        for type in rel_types:
            sub_df = relation_df[relation_df['label']==type]
            new_sub_df = pd.get_dummies(sub_df, columns=['entity_type_1', 'entity_type_2'])

            cor_matrix = pd.DataFrame(columns=entity_types)
            for i, entity_1 in enumerate(entity_types):
                entity_1_col = "entity_type_1_"+entity_1
                row = []
                for j, entity_2 in enumerate(entity_types):
                    entity_2_col = "entity_type_2_"+entity_2
                    try:
                    # if True:
                        n11, n10, n01, n00 = self.co_appearance(new_sub_df[entity_1_col].to_numpy(), new_sub_df[entity_2_col].to_numpy())
                        # value = new_sub_df[entity_1_col].corr(new_sub_df[entity_2_col])

                        value = n11 / (np.sum(new_sub_df[entity_1_col].to_numpy())+np.sum(new_sub_df[entity_2_col].to_numpy(),))
                        # value = (n11*n00 - n10*n01) / (math.sqrt(n11*n01*n10*n00))
                    except:
                        value = 0

                    
                    row.append(value)

                cor_matrix.loc[i] = row
            cor_matrix.index = entity_types

            plt.figure(figsize = (10,10))
            # ax = sns.heatmap(cor_matrix, cmap='Blues_r', annot=False, linewidth=.5)
            ax = sns.heatmap(cor_matrix, annot=False, linewidth=.5)
            plt.xlabel(xlabel="Entity 2")
            plt.ylabel(ylabel="Entity 1")
            plt.title(type)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
            plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
            plt.savefig(saved_name+"/"+type+".png", bbox_inches='tight')
            plt.close('all')

    
if __name__ == "__main__":
    json_path = "experimental_ovp_dataset_v2/mixed_set/mixed_set.json"
    saved_dir = "experimental_ovp_dataset_v2/mixed_set/analysis"
    analysis_data = AnalysisData()
    analysis_data._run_analysis_(json_path, saved_dir)