import os, io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title='Student Risk Predictor', page_icon='🎓', layout='wide')

FEATURES = ['attendance', 'marks', 'assignments', 'study_hours', 'backlogs', 'previous_grade_score']
GRADE_MAP = {'F': 35, 'D': 45, 'C': 55, 'C+': 60, 'B': 68, 'B+': 75, 'A': 85, 'A+': 95}

def make_demo_data(n=240, seed=42):
    rng = np.random.default_rng(seed)
    attendance = rng.integers(35, 101, n)
    marks = np.clip(rng.normal(67, 17, n), 15, 100)
    assignments = np.clip(rng.normal(70, 20, n), 0, 100)
    study_hours = np.clip(rng.normal(8, 4, n), 0, 25)
    backlogs = np.clip(rng.poisson(0.8, n), 0, 6)
    previous_grade_score = np.clip(rng.normal(68, 16, n), 25, 100)
    # Synthetic labels are for demonstration only, not real-world evidence.
    risk_signal = (0.035*(65-attendance) + 0.025*(60-marks) + 0.018*(60-assignments) + 0.35*(4-study_hours) + 0.9*backlogs + 0.018*(60-previous_grade_score) + rng.normal(0, 1.0, n))
    cutoff = np.quantile(risk_signal, 0.67)
    risk = (risk_signal > cutoff).astype(int)
    return pd.DataFrame({'attendance':attendance,'marks':marks,'assignments':assignments,'study_hours':study_hours,'backlogs':backlogs,'previous_grade_score':previous_grade_score,'risk_label':risk})

@st.cache_resource
def train_models():
    df = make_demo_data()
    X, y = df[FEATURES], df['risk_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    models = {
        'Logistic Regression': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))]),
        'Support Vector Machine (SVM)': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', SVC(probability=True, class_weight='balanced', random_state=42))])
    }
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = {'Accuracy':accuracy_score(y_test,pred), 'Precision':precision_score(y_test,pred,zero_division=0), 'Recall':recall_score(y_test,pred,zero_division=0), 'F1-score':f1_score(y_test,pred,zero_division=0), 'Confusion matrix':confusion_matrix(y_test,pred)}
    return models, metrics, df

models, metrics, synthetic_data = train_models()
if 'students' not in st.session_state:
    st.session_state.students = []

st.markdown('''<style>
.block-container{padding-top:1.5rem;padding-bottom:2rem} .hero{padding:1.2rem 1.4rem;border-radius:16px;background:linear-gradient(120deg,#10294b,#1768d5);color:white;margin-bottom:1rem} .hero h1{color:white;margin:0} .hero p{opacity:.9;margin:.35rem 0 0} div[data-testid="stMetric"]{background:#f6f9ff;padding:14px;border-radius:12px;border:1px solid #000000}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🎓 Student Academic Performance & Dropout Risk Predictor</h1><p>Early support through academic indicators · Presentation demo</p></div>', unsafe_allow_html=True)
st.warning('Demo limitation: the model is trained on synthetic sample data generated for this presentation, not on real student records. Predictions and metrics are illustrative and must not be used for actual student decisions.')

with st.sidebar:
    st.title('🎓 StudentRiskPredictor')
    page = st.radio('Navigation', ['Dashboard','Predict Student','Student Records','Model Evaluation','About Project'])
    st.divider()
    st.caption('Built with Python + Scikit-learn')


def predict_one(row):
    x = pd.DataFrame([row])[FEATURES]
    probs = {name: float(model.predict_proba(x)[0][1]) for name, model in models.items()}
    best = max(metrics, key=lambda name: metrics[name]['F1-score'])
    p = probs[best]
    risk = 'High' if p >= .67 else 'Medium' if p >= .34 else 'Low'
    score = 0.4*row['marks'] + 0.2*row['attendance'] + 0.2*row['assignments'] + 0.2*row['previous_grade_score']
    performance = 'Excellent' if score >= 85 else 'Good' if score >= 70 else 'Average' if score >= 55 else 'Poor'
    recommendations=[]
    if row['attendance'] < 75: recommendations.append('Improve attendance and check barriers to attending classes.')
    if row['marks'] < 60: recommendations.append('Arrange subject-wise revision and faculty mentoring.')
    if row['assignments'] < 65: recommendations.append('Set a weekly assignment completion plan.')
    if row['study_hours'] < 6: recommendations.append('Create a realistic, consistent study timetable.')
    if row['backlogs'] > 0: recommendations.append('Plan backlog support sessions and monitor progress.')
    if not recommendations: recommendations.append('Maintain current study habits and review progress regularly.')
    return {'risk':risk, 'risk_probability':p, 'performance_score':score, 'performance':performance, 'model':best, 'recommendations':recommendations}

if page == 'Dashboard':
    students = st.session_state.students
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Students assessed', len(students))
    c2.metric('Average marks', f"{np.mean([s['marks'] for s in students]):.1f}%" if students else '—')
    c3.metric('Medium / High risk', sum(s['prediction']['risk'] in ['Medium','High'] for s in students))
    c4.metric('Models trained', '2')
    st.subheader('Project overview')
    left,right=st.columns(2)
    with left:
        st.markdown('#### Prediction workflow')
        st.markdown('1. Enter academic indicators\n2. Preprocess and scale the features\n3. Compare Logistic Regression and SVM\n4. Estimate risk and performance category\n5. Suggest supportive interventions')
    with right:
        st.markdown('#### Current records')
        if students:
            frame=pd.DataFrame([{'ID':s['id'],'Name':s['name'],'Performance':s['prediction']['performance'],'Risk':s['prediction']['risk'],'Marks':s['marks']} for s in students])
            st.dataframe(frame, use_container_width=True, hide_index=True)
        else:
            st.info('No student records yet. Open **Predict Student** to try a sample case, or load demo records below.')
    if st.button('Load 5 demo student records'):
        samples=[('S001','Aditi Sharma',92,88,90,12,0,'A'),('S002','Rohan Mehta',78,74,72,8,0,'B+'),('S003','Priya Nair',63,56,60,5,1,'C'),('S004','Arjun Rao',48,38,42,3,3,'D'),('S005','Neha Kulkarni',85,78,75,10,0,'B+')]
        st.session_state.students=[]
        for sid,name,att,marks,assign,hrs,backs,grade in samples:
            row={'attendance':att,'marks':marks,'assignments':assign,'study_hours':hrs,'backlogs':backs,'previous_grade_score':GRADE_MAP[grade]}
            st.session_state.students.append({'id':sid,'name':name,**row,'prediction':predict_one(row)})
        st.rerun()
elif page == 'Predict Student':
    st.subheader('Predict a student')
    with st.form('predict_form'):
        a,b=st.columns(2)
        with a:
            sid=st.text_input('Student ID', value=f'S{len(st.session_state.students)+1:03d}')
            name=st.text_input('Student name')
            attendance=st.slider('Attendance (%)',0,100,75)
            marks=st.slider('Internal marks (%)',0,100,65)
            assignments=st.slider('Assignment score (%)',0,100,70)
        with b:
            study_hours=st.slider('Study hours per week',0,40,8)
            backlogs=st.number_input('Number of backlogs',0,20,0)
            grade=st.selectbox('Previous semester grade',list(GRADE_MAP),index=5)
            st.caption('Enter approximate academic values for demonstration.')
        submitted=st.form_submit_button('🔎 Predict & Save', type='primary', use_container_width=True)
    if submitted:
        if not sid.strip() or not name.strip(): st.error('Please enter both Student ID and name.')
        elif any(s['id'].lower()==sid.strip().lower() for s in st.session_state.students): st.error('That Student ID already exists.')
        else:
            row={'attendance':attendance,'marks':marks,'assignments':assignments,'study_hours':study_hours,'backlogs':backlogs,'previous_grade_score':GRADE_MAP[grade]}
            st.session_state.students.append({'id':sid.strip(),'name':name.strip(),**row,'prediction':predict_one(row)})
            st.success('Prediction created and student saved.')
            st.rerun()
    st.markdown('#### Quick example')
    st.caption('Try a student with 45% attendance, 40 marks, low assignment completion, 3 study hours and 3 backlogs to see how the model responds.')
elif page == 'Student Records':
    st.subheader('Student records')
    students=st.session_state.students
    if not students: st.info('No records yet. Load demo records from Dashboard or add one on Predict Student.')
    else:
        search=st.text_input('Search by ID or name')
        shown=[s for s in students if search.lower() in s['id'].lower() or search.lower() in s['name'].lower()]
        df=pd.DataFrame([{'ID':s['id'],'Name':s['name'],'Attendance %':s['attendance'],'Marks %':s['marks'],'Assignments %':s['assignments'],'Study hrs/wk':s['study_hours'],'Backlogs':s['backlogs'],'Performance':s['prediction']['performance'],'Dropout risk':s['prediction']['risk']} for s in shown])
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button('⬇ Download CSV',df.to_csv(index=False).encode('utf-8'),'student_risk_records.csv','text/csv')
        choices={f"{s['id']} — {s['name']}":s['id'] for s in shown}
        if choices:
            chosen=st.selectbox('View student report',list(choices))
            s=next(x for x in students if x['id']==choices[chosen]); p=s['prediction']
            st.markdown(f"### {s['name']} ({s['id']})")
            c1,c2,c3=st.columns(3); c1.metric('Performance',p['performance']); c2.metric('Dropout risk',p['risk']); c3.metric('Performance score',f"{p['performance_score']:.1f}/100")
            st.progress(min(1,p['risk_probability']),text=f"Illustrative model risk probability: {p['risk_probability']:.0%}")
            st.write('**Recommended support**')
            for rec in p['recommendations']: st.markdown(f'- {rec}')
        if st.button('Clear all student records',type='secondary'):
            st.session_state.students=[]; st.rerun()
elif page == 'Model Evaluation':
    st.subheader('Model evaluation (synthetic test split)')
    st.caption('Both models were trained on generated sample labels. These scores only demonstrate the evaluation workflow; they are not evidence of real-world prediction accuracy.')
    metric_rows=[]
    for name,m in metrics.items(): metric_rows.append({'Model':name,**{k:round(v,3) for k,v in m.items() if k!='Confusion matrix'}})
    st.dataframe(pd.DataFrame(metric_rows),use_container_width=True,hide_index=True)
    for name,m in metrics.items():
        with st.expander(f'{name} — confusion matrix'):
            st.write('Rows = actual class [not flagged, flagged]; columns = predicted class [not flagged, flagged].')
            st.dataframe(pd.DataFrame(m['Confusion matrix'],index=['Actual not flagged','Actual flagged'],columns=['Predicted not flagged','Predicted flagged']))
    st.markdown('#### About SMOTE')
    st.info('Your report describes SMOTE for imbalanced datasets. This demo uses a synthetic dataset with roughly balanced labels, so SMOTE is not applied. For a real dataset, apply SMOTE only to the training fold (inside cross-validation/pipeline) and compare results carefully.')
elif page == 'About Project':
    st.subheader('About the project')
    st.write('The Student Academic Performance & Dropout Risk Predictor aims to identify students who may need additional academic support, using attendance, marks, assignment completion, study hours, backlogs, and previous grades.')
    st.markdown('**Technologies:** Python, Pandas, NumPy, Scikit-learn (Logistic Regression and SVM).')
    st.markdown('**Methodology:** sample data → preprocessing → train/test split → model training → evaluation (accuracy, precision, recall, F1-score, confusion matrix) → predictions and support recommendations.')
    st.warning('Limitations: no real dataset, no validated labels, and no real-world performance claim. Do not use the demo to make actual decisions about students. A real deployment needs consent, privacy safeguards, representative data, expert review, and validation.')
    st.markdown('**Presentation line:** “Because a real dataset was not available for this prototype, I generated synthetic sample data to demonstrate the end-to-end machine-learning workflow. The results are illustrative and the model must be retrained and validated on real institutional data before use.”')
