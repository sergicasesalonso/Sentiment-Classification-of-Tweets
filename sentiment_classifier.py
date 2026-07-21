
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, cohen_kappa_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
import pickle




# 1. LOAD DATA
print("1. LOADING DATA")

# crowdsourced_train and gold_train cover the SAME tweets with different labels.
# test.csv contains entirely different tweets with trusted (gold) labels.
crowdsourced = pd.read_csv('HW3/crowdsourced_train.csv', sep='\t', encoding='utf-8')
gold         = pd.read_csv('HW3/gold_train.csv',         sep='\t', encoding='utf-8')
test         = pd.read_csv('HW3/test.csv',               sep='\t', encoding='utf-8')

print(f"Crowdsourced: {len(crowdsourced)}, Gold: {len(gold)}, Test: {len(test)}")


# 2. CLEAN CROWDSOURCED LABELS
print("2. DATA EXPLORATION & CLEANING")

print("Raw crowdsourced unique labels:", sorted(crowdsourced['sentiment'].unique()))

#lowercase and strip surrounding whitespace to reduce variant count.
crowdsourced['sentiment'] = crowdsourced['sentiment'].str.lower().str.strip()

#define sets of known misspellings for each canonical class.
NEG = {'negative', 'negtaive', 'nedative', 'negayive'}
POS = {'positive', 'postive',  'positie',  'postitive', 'positve', 'npositive'}
NEU = {'neutral',  'nuetral',  'netural',  'neutrla',   'neutrall', 'neugral',
       'neural',   'nutral',   'neutal',   'neutra l',  'netutral', 'neutral?',
       '_x0008_neutral'}

def norm(lbl):
    """Map a raw label string to one of the three canonical classes, or None if unresolvable."""
    if lbl in NEG: return 'negative'
    if lbl in POS: return 'positive'
    if lbl in NEU: return 'neutral'
    return None   # will be dropped below

crowdsourced['sentiment'] = crowdsourced['sentiment'].apply(norm)

dropped = crowdsourced['sentiment'].isna().sum()
print(f"Unresolvable labels dropped: {dropped}")

# drop any rows whose label could not be resolved
keep_mask    = crowdsourced['sentiment'].notna()
crowdsourced = crowdsourced[keep_mask].reset_index(drop=True)
gold_aligned = gold[keep_mask.values].reset_index(drop=True)  # same rows as crowdsourced

print("Labels after cleaning:", sorted(crowdsourced['sentiment'].unique()))

# Compute per-split label counts 
cs_dist   = crowdsourced['sentiment'].value_counts()
gold_dist = gold['sentiment'].value_counts()
test_dist = test['sentiment'].value_counts()
print("\nCrowdsourced dist:\n", cs_dist)
print("\nGold dist:\n", gold_dist)


# 3. INTER-ANNOTATOR AGREEMENT

print("3. INTER-ANNOTATOR AGREEMENT")

# We treat the crowdsourced set and the gold set as two separate annotators
# who labelled the same tweets. This lets us quantify how much they agree.
cs_labels   = crowdsourced['sentiment'].values
gold_labels = gold_aligned['sentiment'].values

#fraction of tweets where both annotators chose the same label.
raw_acc = np.mean(cs_labels == gold_labels)


kappa = cohen_kappa_score(cs_labels, gold_labels)

print(f"Raw agreement : {raw_acc:.4f} ({raw_acc*100:.1f}%)")
print(f"Cohen's Kappa : {kappa:.4f}")

# Cross-tabulation shows which pairs of (crowdsourced, gold) labels co-occur
merged = pd.DataFrame({'crowdsourced': cs_labels, 'gold': gold_labels})
ct = pd.crosstab(merged['crowdsourced'], merged['gold'])
print("\nCross-tabulation:\n", ct)



# 4. VISUALISATIONS

labels_order = ['positive', 'neutral', 'negative'] 

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# bar chart comparing crowdsourced vs gold label counts —
x = np.arange(3); w = 0.35
cs_c   = [int(cs_dist.get(l, 0))   for l in labels_order]
gold_c = [int(gold_dist.get(l, 0)) for l in labels_order]
axes[0].bar(x - w/2, cs_c,   w, label='Crowdsourced', color='steelblue')
axes[0].bar(x + w/2, gold_c, w, label='Gold',         color='coral')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels_order)
axes[0].set_title('Label Distribution'); axes[0].set_ylabel('Count')
axes[0].legend()

#  heatmap of the annotator cross-tabulation (reindexed to fixed order) —
ct3 = ct.reindex(index=labels_order, columns=labels_order, fill_value=0)
sns.heatmap(ct3, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title(f'Annotator Agreement\nκ = {kappa:.3f}')
axes[1].set_xlabel('Gold'); axes[1].set_ylabel('Crowdsourced')

#label distribution in the test set —
axes[2].bar(labels_order,
            [int(test_dist.get(l, 0)) for l in labels_order],
            color=['seagreen', 'steelblue', 'tomato'])
axes[2].set_title('Test Distribution'); axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('HW3/data_exploration.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: data_exploration.png")


# 5. MODEL BUILDING

print("4. MODEL TRAINING")

def build_pipeline():
    
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=50000,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
    )
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        random_state=42,
    )
    return Pipeline([('tfidf', tfidf), ('clf', clf)])


def evaluate(name, y_true, y_pred):
    """Print accuracy + per-class precision/recall/F1 and return accuracy."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[{name}] Test accuracy: {acc:.4f}")
    # classification_report shows precision, recall, F1 for each class and macro/weighted averages
    print(classification_report(y_true, y_pred))
    return acc


#  always predict the most common class, Any model that beats this is doing something genuinely useful.
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(gold['text'], gold['sentiment'])
dummy_acc = accuracy_score(test['sentiment'], dummy.predict(test['text']))
print(f"Baseline (majority class) test accuracy: {dummy_acc:.4f}")


# Model 1: trained on CROWDSOURCED labels 
pipe_cs = build_pipeline()

# 5-fold cross-validation on the training data gives a estimate of generalisation before  the test set.
cv_cs = cross_val_score(pipe_cs, crowdsourced['text'], crowdsourced['sentiment'],
                        cv=5, scoring='accuracy')
print(f"\nCrowdsourced 5-fold CV: {cv_cs.mean():.4f} ± {cv_cs.std():.4f}")

# Retrain on the full crowdsourced training set, then evaluate on test.
pipe_cs.fit(crowdsourced['text'], crowdsourced['sentiment'])
preds_cs = pipe_cs.predict(test['text'])
acc_cs   = evaluate("Crowdsourced", test['sentiment'], preds_cs)

# confusion_matrix rows = true labels, columns = predicted labels.
cm_cs = confusion_matrix(test['sentiment'], preds_cs, labels=labels_order)


# Model 2: trained on GOLD labels 
pipe_gold = build_pipeline()

cv_gold = cross_val_score(pipe_gold, gold['text'], gold['sentiment'],
                          cv=5, scoring='accuracy')
print(f"\nGold 5-fold CV: {cv_gold.mean():.4f} ± {cv_gold.std():.4f}")

pipe_gold.fit(gold['text'], gold['sentiment'])
preds_gold = pipe_gold.predict(test['text'])
acc_gold   = evaluate("Gold", test['sentiment'], preds_gold)
cm_gold    = confusion_matrix(test['sentiment'], preds_gold, labels=labels_order)



# 6. CONFUSION MATRICES

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

for ax, cm, title in zip(axes2,
                          [cm_cs, cm_gold],
                          ['Crowdsourced Training', 'Gold Training']):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=labels_order, yticklabels=labels_order)
    ax.set_title(f'Confusion Matrix – {title}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')

plt.tight_layout()
plt.savefig('HW3/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: confusion_matrices.png")


# 7. TOP FEATURES PER CLASS
# Each row of clf.coef_ corresponds to one class. Higher weight = stronger
# association with that class
vocab = np.array(pipe_gold.named_steps['tfidf'].get_feature_names_out())
clf   = pipe_gold.named_steps['clf']

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
colors_map = {'negative': 'tomato', 'neutral': 'steelblue', 'positive': 'seagreen'}

for ax, (i, cls) in zip(axes3, enumerate(clf.classes_)):
    # argsort returns indices in ascending order; take last 10 for highest weights.
    idx = np.argsort(clf.coef_[i])[-10:][::-1]  
    ax.barh(vocab[idx][::-1], clf.coef_[i][idx][::-1],
            color=colors_map.get(cls, 'gray'), alpha=0.8)
    ax.set_title(f'Top features: {cls}')
    ax.set_xlabel('Weight')

plt.tight_layout()
plt.savefig('HW3/top_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: top_features.png")



# 8. SUMMARY

print("SUMMARY")
print(f"{'Model':<30} {'CV Acc':>10} {'Test Acc':>10}")
print("-" * 52)
print(f"{'Baseline (majority class)':<30} {'—':>10} {dummy_acc:>10.4f}")
print(f"{'Logistic Reg (Crowdsourced)':<30} {cv_cs.mean():>10.4f} {acc_cs:>10.4f}")
print(f"{'Logistic Reg (Gold)':<30} {cv_gold.mean():>10.4f} {acc_gold:>10.4f}")

# Persist all key results so the report-generation script can load them
# without having to re-run the entire training pipeline.
with open('HW3/results.pkl', 'wb') as f:
    pickle.dump({
        'dummy_acc': dummy_acc,
        'cv_cs':     cv_cs,   'acc_cs':  acc_cs,
        'cv_gold':   cv_gold, 'acc_gold': acc_gold,
        'kappa':     kappa,   'raw_acc':  raw_acc,
        'cs_dist':   dict(cs_dist),
        'gold_dist': dict(gold_dist),
        'test_dist': dict(test_dist),
        'cm_cs':     cm_cs,
        'cm_gold':   cm_gold,
        'labels_order': labels_order,
    }, f)

print("Done. All artefacts saved to HW3/")