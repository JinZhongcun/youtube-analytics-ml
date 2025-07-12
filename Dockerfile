FROM python:3.9

WORKDIR /work

RUN pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost

COPY . /work/

CMD ["python", "comprehensive_model_comparison.py"]