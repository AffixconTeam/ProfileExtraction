import streamlit as st
import numpy as np
import pandas as pd
import datetime
import gender_guesser.detector as gender
from kneed import KneeLocator 
d = gender.Detector()
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import plotly.express as px
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import ast 


st.set_page_config(layout='wide')
# st.title(':red[DataZoo Profile Extraction]')
st.markdown("<h1 style='text-align: center; color: blue;'>DataZoo Profile Extraction</h1>", unsafe_allow_html=True)


#---------------------------------------------Functions --------------------------------------------------------------------
def calculate_match(row, profile):
    total_fields = len(profile)  -2 
    matches = 0
    
    for field in profile:
        if field == 'Profile' or field =='Cluster':
            continue
        if (profile[field] != '' and row[field]!='') and row[field] == profile[field]:
            matches += 1
    return (matches / total_fields) * 100

def multi_calculate_match(row, profile):
    total_fields = len(profile) - 2  
    matches = 0
    for field in profile:
        if field == 'Profile':
            continue

        if profile.get(field) and row.get(field):
            if isinstance(profile[field], str) and ',' in profile[field]:
                profile_values = [val.strip() for val in profile[field].split(', ')]
            else:
                profile_values = [profile[field]] 

            if str(row[field]) in profile_values:
                matches += 1
    return (matches / total_fields) * 100

def age(dob):
    born_year = pd.to_datetime(dob).date().year
    current_year = datetime.datetime.now().year
    age = current_year - born_year
    return age

def predict_gender(row):
    if pd.isna(row['Gender']): 
        for name_column in ['FirstName', 'SecondName', 'Surname']:
            name = row.get(name_column)
            if pd.notna(name): 
                predicted_gender = d.get_gender(name)
                if predicted_gender in ['male', 'female']:
                    return predicted_gender
    return row['Gender']

@st.cache_data
def preprocss(df):
    
    #------------------------------Age-------------------------------------------
    df['Age'] = df['DateOfBirth'].apply(age)
    age_ranges = {
    "1-5": (1, 5),
    "6-10": (6, 10),
    "11-15": (11, 15),
    "16-20": (16, 20),
    "21-25": (21, 25),
    "26-30": (26, 30),
    "31-35": (31, 35),
    "36-40": (36, 40),
    "41-45": (41, 45),
    "46-50": (46, 50),
    "51-55": (51, 55),
    "56-60": (56, 60),
    "61-65": (61, 65),
    "66-70": (66, 70),
    "71-75": (71, 75),
    "76-80": (76, 80),
    "81-85": (81, 85),
    "86-90": (86, 90),
    "91-95": (91, 95),
    "Over 95": (96, float('inf'))}
    def get_age_range(age):
        for label, (lower, upper) in age_ranges.items():
            if lower <= age <= upper:
                return label
        return "Unknown" 
    
    df['AgeRangeNew'] = df['Age'].apply(get_age_range)
    
    #------------------------------PropertyValue-------------------------------------------
    bins = [0, 100000, 300000, 500000, 1000000, np.inf]  
    labels = ['0-100k', '100k-300k', '300k-500k', '500k-1M', '1M+']  
    df['PropertyValue'] = pd.to_numeric(df['PropertyValue'], errors='coerce')
    df['PropertyValueRange'] = pd.cut(df['PropertyValue'], bins=bins, labels=labels)
    df['PropertyValueRange'] = df['PropertyValueRange'].cat.add_categories(['']).fillna('')
    
    #------------------------------Gender-------------------------------------------
    df['Gender'] = df['Gender'].replace({'Lady':'Female', 'Fath':'Male'})
    df['GenderNew'] = df.apply(predict_gender, axis=1)
    df['GenderNew'] = df['GenderNew'].replace({'female':'Female','male':'Male','mostly_female':'Female','mostly_male':'Male'})

    #------------------------------HomeOwner-------------------------------------------
    df['NewHomeOwner'] = df['NewHomeOwner'].fillna("renters")
    df['NewHomeOwner'] = df['NewHomeOwner'].replace({1.0:"HomeOwner"})
    
    df = df.dropna(subset=['IncomeRange','GenderNew','AgeRangeNew', 'ProfileType', 'DirShr_Category', 'NewHomeOwner', "PropertyValueRange",'Age','PropertyValue','Occupation','ProfileDescription'],how='all')
    analysis_column = ['URN','GenderNew', 'IncomeRange','AgeRangeNew', 'ProfileType', 'DirShr_Category', 'NewHomeOwner', "PropertyValueRange",'Age','PropertyValue','Occupation','ProfileDescription']
    # df_analysis = df.copy()
    df = df[analysis_column]
    
    df = df.replace(np.nan, "")
    df['Concatenated'] = df[['IncomeRange','AgeRangeNew', 'ProfileType', 'DirShr_Category', 'NewHomeOwner', "PropertyValueRange","Occupation"]].astype(str).agg(' '.join, axis=1)
    df['Concatenated'] = df['Concatenated'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df.Concatenated = df.Concatenated.str.lower()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    return df

@st.cache_data
def normalization(df):

    encoded_df = preprocss(df).fillna("").copy()
    # LE=LabelEncoder()
    cols_encode = ['IncomeRange','GenderNew','AgeRangeNew', 'ProfileType', 'DirShr_Category', 'NewHomeOwner', "PropertyValueRange"]
    # for i in cols_encode:
    #     encoded_df[i]=encoded_df[[i]].apply(LE.fit_transform)
    income_order = {'': 9, '<20,000': 1, '>20,000': 2, '20,001 - 30,000':3, '30,001 - 50,000':4, '50,001 - 70,000':5, '70,001 - 100,000':6, '100,001 - 150,000':7, '150,001 - 200,000':8}
    age_order = {
        'Unknown':1,'1-5':2,'6-10': 3,'11-15': 4,'16-20':5,'21-25': 6,'26-30': 7,'31-35': 8,'36-40': 9,'41-45': 10,'46-50': 11,'51-55': 12,'56-60': 13,'61-65': 14,'66-70': 15,
        '71-75': 16,'76-80': 17,'81-85': 18,'86-90': 19,'91-95': 20,'Over 95':21}
    dir_shr_order = {'Director & Shareholder':4, '':1, 'Director':2, 'Shareholder':3}
    profile_category_order = {
    '':1,'A1': 2,'A2': 3,'A3': 4,'A4': 5,'A5': 6,'B1': 7,'B2': 8,'B3': 9,'B4':10,'B5': 11,'C1': 12,'C2': 13,'C3': 14,'C4': 15,'C5': 16,'D1': 17,'D2': 18,'D3': 19,'D4': 20,
    'E1': 21,'E2': 22,'E3': 23,'E4': 24,'E5': 25,'E6': 26,'F1': 27,'F2': 28}
    home_owner_order = {'':1,'renters':2,'HomeOwner':3}
    gender_order = {'Male':3,'Female':2,'':1}
    property_value_range_order = {'0-100k':2, '100k-300k':3, '300k-500k':4, '500k-1M':5, '1M+':6, "":1}


    df_encoded= encoded_df.copy()

    df_encoded['IncomeRange'] = encoded_df['IncomeRange'].map(income_order)
    df_encoded['AgeRangeNew'] = encoded_df['AgeRangeNew'].map(age_order)
    df_encoded['DirShr_Category'] = encoded_df['DirShr_Category'].map(dir_shr_order)
    df_encoded['ProfileType'] = encoded_df['ProfileType'].map(profile_category_order)
    df_encoded['GenderNew'] = encoded_df['GenderNew'].map(gender_order)
    df_encoded['NewHomeOwner'] = encoded_df['NewHomeOwner'].map(home_owner_order)
    df_encoded['PropertyValueRange'] = encoded_df['PropertyValueRange'].map(property_value_range_order)

    df_encoded = df_encoded[cols_encode]
    # #Scaling
    # scaler = StandardScaler()
    # scaler.fit(df_encoded[cols_encode])
    # scaled_df = pd.DataFrame(scaler.transform(df_encoded[cols_encode]),columns= df_encoded[cols_encode].columns )
    # print("All features are now scaled")
    
    return df_encoded

@st.cache_data
def Dimensional_Reduction(df):
    scaled_df = normalization(df)
    pca = PCA(n_components=3)
    pca.fit(scaled_df)
    PCA_df = pd.DataFrame(pca.transform(scaled_df), columns=(["col1","col2", "col3"]))
    
    return PCA_df

def Threshold(column):
    test1 = breakdowns_all[breakdowns_all['Category']==column]
    # val = (test1[f'Group_{i}_Relative_Percentage'].tolist() i for i in range(optimal_clusters))
    # val = [test1[f'Group_{i}_Relative_Percentage'].tolist() for i in range(1, optimal_clusters + 1)]
    # Filter columns ending with '_Relative_Percentage'
    relative_cols = [col for col in test1.columns if col.endswith('_Relative_Percentage')]
    val = test1[relative_cols].values.tolist()  
    thresold = np.mean(val)+0.5*np.std(val)
    return thresold

def remove_empty_elements_from_string(lst_str):
    try:
        lst = ast.literal_eval(lst_str)
        return str([item for item in lst if not item.startswith("empty_")])
    except (ValueError, SyntaxError):
        return lst_str
#------------------------------------------------------ End of Functions --------------------------------------------------
        
file = st.sidebar.file_uploader('Choose a File')
if file is not None:
    data = pd.read_csv(file, encoding = 'latin1' )
    dilworth = data[data['File Name']=='Dilworth']

    scaled_df = normalization(dilworth)
    PCA_df = Dimensional_Reduction(dilworth)
    # st.write(PCA_df.head())
    
    Elbow_M = KElbowVisualizer(KMeans(random_state=42), k=10)
    Elbow_M.fit(Dimensional_Reduction(dilworth))
    Elbow_M.ax.set_title("Elbow Method for Optimal Clusters")
    col1,col2,col3 = st.columns((3))
    with col1:
        with st.expander(":red[Show Optimal Cluster Selection]"):
            st.pyplot(Elbow_M.ax.figure)
    
    inertia_list = []
    cluster_range = range(2, 10)  

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(PCA_df)
        
        inertia_list.append(kmeans.inertia_)

    kl = KneeLocator(cluster_range, inertia_list, curve="convex", direction="decreasing")
    optimal_clusters = kl.elbow
    # st.write(f"The optimal number of clusters : {optimal_clusters}")


    AC = AgglomerativeClustering(n_clusters=optimal_clusters)
    yhat_AC = AC.fit_predict(PCA_df)
    PCA_df["Clusters"] = yhat_AC+1
    dilworth_preprocessed = preprocss(dilworth)
    dilworth_preprocessed["Clusters"] = yhat_AC+1

    fig = px.scatter_3d(
        PCA_df, 
        x="col1", 
        y="col2", 
        z="col3", 
        color='Clusters', 
        title="All Groups as clusters"
    )

    with col2:
        with st.expander(":red[Show 3d Cluster Separation]"):
            st.plotly_chart(fig)

    with col3:
        pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
        fig, ax = plt.subplots(figsize=(8, 6))
        pl = sns.countplot(x=dilworth_preprocessed["Clusters"], palette=pal, ax=ax)
        pl.set_title("Distribution Of The Clusters")
        total = len(dilworth_preprocessed["Clusters"])
        for patch in ax.patches:
            height = patch.get_height()
            percentage = f"{(height / total) * 100:.1f}%"
            ax.annotate(percentage, (patch.get_x() + patch.get_width() / 2, height),
                        ha="center", va="bottom", fontsize=10)

        with st.expander(":red[Show Distribution of Clusters]"):
            st.pyplot(fig)


    # ----------------------- Sample Breakdown ---------------------------------------
    cols_del = ['ProfileType','IncomeRange', 'DirShr_Category', 'NewHomeOwner', 'GenderNew', 'AgeRangeNew', 'PropertyValueRange']
    dfs = []
    for col in cols_del:
        count_df = pd.DataFrame(dilworth_preprocessed[col].value_counts())
        percent_df = pd.DataFrame(dilworth_preprocessed[col].value_counts(normalize=True) * 100)
        combined_df = pd.concat([count_df, percent_df], axis=1)
        combined_df['Category'] = col
        combined_df.columns = ['Count', 'Percentage','Category']
        combined_df.index = combined_df.index.map(lambda x: f"empty_{col}" if x == "" else x)
        dfs.append(combined_df)

    final_df = pd.concat(dfs, axis=0)
    final_df.columns = ['Sample_Count', 'Sample_Percentage','Category']

    # st.write(final_df)
    # ----------------------- Sample with all groups count Breakdown ---------------------------------------
    dfs_all = []
    for cluster in sorted(dilworth_preprocessed.Clusters.unique().tolist()):
        dfs = []
        profile_df = dilworth_preprocessed[dilworth_preprocessed['Clusters'] == cluster][cols_del]

        for col in cols_del:
            count_df = pd.DataFrame(profile_df[col].value_counts())
            count_df.columns = [f'Group_{cluster}_Count']
            count_df.index = count_df.index.map(lambda x: f"empty_{col}" if x == "" else x)
            dfs.append(count_df)
        profile_df_values = pd.concat(dfs, axis=0)
        profile_df_values = profile_df_values.reindex(final_df.index, fill_value=0)
        dfs_all.append(profile_df_values)
        
    dfs_all = pd.concat(dfs_all,axis=1)

    breakdowns_all = pd.concat([final_df,dfs_all],axis=1)
    for i in range(1, optimal_clusters+1):
        breakdowns_all[f"Group_{i}_Percentage"] = (breakdowns_all[f"Group_{i}_Count"] / breakdowns_all['Sample_Count']) * 100
        breakdowns_all[f"Group_{i}_Relative_Percentage"] = (breakdowns_all[f"Group_{i}_Percentage"] * breakdowns_all['Sample_Percentage']) /100
    
    # st.write(breakdowns_all)

    # test1 = breakdowns_all[breakdowns_all['Category']=='ProfileType']
    # # val = (test1[f'Group_{i}_Relative_Percentage'].tolist() i for i in range(optimal_clusters))
    # val = [test1[f'Group_{i}_Relative_Percentage'].tolist() for i in range(1, optimal_clusters + 1)]
    profile_thresold = Threshold('ProfileType')
    # st.write(profile_thresold)

    # -----------------------------------------------Profile Type extraction --------------------------------------
    test = breakdowns_all[breakdowns_all['Category'] == 'ProfileType']
    relaive_cols = [f'Group_{i}_Relative_Percentage' for i in range(1,optimal_clusters+1) ]
    data = test[relaive_cols][test[relaive_cols]>profile_thresold].dropna(how='all').fillna(0).idxmax(axis=1).sort_values().to_dict()
    group_dict = {}
    for value, group in data.items():
        split_group = group.split('_', 2)
        group_prefix = '_'.join(split_group[:2])
        if group not in group_dict:
            group_dict[group] = [] 
        group_dict[group].append(value)  
    new_dict = {}

    for key, value in group_dict.items():
        new_key = '_'.join(key.split('_', 2)[:2])  
        new_dict[new_key] = value

    # st.write(new_dict)

    income_threshold = Threshold('IncomeRange')
    # st.write('profile_threshold previous', profile_thresold)

# ------------------------------------------- select other demog --------------------
    # results = {}
    # for i in range(1, optimal_clusters + 1):
    #     column = f"Group_{i}_Relative_Percentage"
        
    #     # Initialize the dictionary for the current group
    #     results[f"Group_{i}"] = {}

    #     filtered = breakdowns_all.copy()

    #     for col in cols_del:
    #         filtered_range = filtered[(filtered["Category"] == col) & (filtered[column] > Threshold(col))]
            
    #         # Check if filtered_range is not empty
    #         if not filtered_range.empty:
    #             results[f"Group_{i}"][col] = str(filtered_range.index.tolist())

    results = {}
    for i in range(1, optimal_clusters + 1):
        column = f"Group_{i}_Relative_Percentage"
        filtered = breakdowns_all.copy()  
        filtered = filtered[
            ((filtered["Category"] == "IncomeRange") & (filtered[column] > Threshold('IncomeRange'))) | 
            ((filtered["Category"] != "IncomeRange") & (filtered[column] > Threshold('ProfileType')))]
        filtered = filtered.sort_values(by=column, ascending=False)
        grouped = filtered.groupby("Category").apply(lambda x: str(x.index.tolist()))
        results[f"Group_{i}"] = grouped.to_dict()
             
    # st.write(results)

    profiles_selected = pd.DataFrame(results).fillna("")
    for i in range(1, optimal_clusters + 1):
        profiles_selected.loc['ProfileType',f'Group_{i}'] = ''
    for i,j in new_dict.items():
        profiles_selected.loc['ProfileType',i] = str(j)

    profiles_selected = profiles_selected.applymap(remove_empty_elements_from_string)
    profiles_selected = profiles_selected.apply(lambda x: x.replace(r"[\[\]']", '', regex=True))

    # st.dataframe(profiles_selected,width=1500)
    cluster_percentages = np.ceil(dilworth_preprocessed['Clusters'].value_counts(normalize=True) * 100)
    cluster_percentages_df = cluster_percentages.to_frame().T  
    cluster_percentages_df.index = ["Cluster_percentage"]    
    cluster_percentages_df.columns = [f'Group_{i}' for i in cluster_percentages.index]
    profiles_selected = pd.concat([profiles_selected, cluster_percentages_df], axis=0).T
    # profiles_selected = profiles_selected.T.sort_values('Cluster_percentage', ascending=False)
    # profiles_selected.index = [f'Group_{i}' for i in range(1,len(cluster_percentages)+1)]
    profiles_selected = profiles_selected.T
    st.dataframe(profiles_selected,width=1700)




    #------------------------------------------------------------Profile Selection ----------------------------
    # for col in breakdowns_all.columns:
    groups = st.multiselect("Select Groups", profiles_selected.columns, default=profiles_selected.columns)
    selected_columns = ['Sample_Count', 'Sample_Percentage','Category'] + [col for col in breakdowns_all.columns if any(group in col for group in groups)]

    breakdowns_all = breakdowns_all[selected_columns]
    # st.dataframe(breakdowns_all,width=1500)
    # st.write(breakdowns_all.columns.tolist())
    # groups_rename = [f'Group_{i+1}' for i in range(len(selected_columns))]

    # selected_column_indices = [i for i, col in enumerate(breakdowns_all.columns) if col in selected_columns]
    # new_column_names = [f"Group_{i+1}_" for i in range(len(selected_columns))]
    # for idx, new_col in zip(selected_column_indices, new_column_names):
    #     breakdowns_all.columns.values[idx] = new_col


    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------Profile Type extraction --------------------------------------
    test = breakdowns_all[breakdowns_all['Category'] == 'ProfileType']
    # relaive_cols = [f'Group_{i}_Relative_Percentage' for i in range(1,len(selected_columns)+1) ]
    # relative_cols = [i for i in breakdowns_all.columns.str.strip() if i.endswith('_Relative_Percentage')]
    relaive_cols = [col for col in breakdowns_all.columns if col.endswith('_Relative_Percentage')]
    # st.write(test.columns)
    profile_thresold = Threshold('ProfileType')
    # st.write('profile_threshold new', profile_thresold)
    
    data = test[relaive_cols][test[relaive_cols]>profile_thresold].dropna(how='all').fillna(0).idxmax(axis=1).sort_values().to_dict()
    group_dict = {}
    for value, group in data.items():
        split_group = group.split('_', 2)
        group_prefix = '_'.join(split_group[:2])
        if group not in group_dict:
            group_dict[group] = [] 
        group_dict[group].append(value)  
    new_dict = {}

    for key, value in group_dict.items():
        new_key = '_'.join(key.split('_', 2)[:2])  + '_Relative_Percentage'
        new_dict[new_key] = value

    # st.write(new_dict)

    income_threshold = Threshold('IncomeRange')
    # st.write('income_threshold_new',income_threshold)

    results = {}
    for column in relaive_cols:
        # column = f"Group_{i}_Relative_Percentage"
        filtered = breakdowns_all.copy()  
        filtered = filtered[
            ((filtered["Category"] == "IncomeRange") & (filtered[column] > Threshold('IncomeRange'))) | 
            ((filtered["Category"] != "IncomeRange") & (filtered[column] > Threshold('ProfileType')))]
        filtered = filtered.sort_values(by=column, ascending=False)
        grouped = filtered.groupby("Category").apply(lambda x: str(x.index.tolist()))
        results[column] = grouped.to_dict()
             

    profiles_selected = pd.DataFrame(results).fillna("")
    for i in relaive_cols:
        profiles_selected.loc['ProfileType',i] = ''
    for i,j in new_dict.items():
        profiles_selected.loc['ProfileType',i] = str(j)

    profiles_selected = profiles_selected.applymap(remove_empty_elements_from_string)
    profiles_selected = profiles_selected.apply(lambda x: x.replace(r"[\[\]']", '', regex=True))

    new_profiles_selected = profiles_selected[relaive_cols]
    rename_cols = [f'Group_{i}' for i in range(1,len(results)+1)]
    new_profiles_selected.columns = rename_cols
    new_cluster_percentages_df = cluster_percentages_df[groups]
    new_cluster_percentages_df.columns = rename_cols
    new_profiles_selected = pd.concat([new_profiles_selected, new_cluster_percentages_df], axis=0)
    # st.dataframe(new_profiles_selected,width=1500)
    new_profiles_selected = new_profiles_selected.T.sort_values('Cluster_percentage',ascending=False)
    new_profiles_selected.index = rename_cols
    # for col in breakdowns_all.columns:\

    st.dataframe(new_profiles_selected.T,width=1500)

