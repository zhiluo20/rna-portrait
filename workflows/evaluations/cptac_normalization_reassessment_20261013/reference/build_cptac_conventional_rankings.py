"""Apply the fixed conventional/metadata definitions without reading qrels."""
import gzip
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parent
COHORT=ROOT/'paired_candidates/CPTAC_PanCancer'
INPUT=COHORT/'conventional_v1'
OUT=COHORT/'conventional_rankings_v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def query_score(percentiles,q):
    columns=[percentiles[x] for x in q['positive_features']]+[1-percentiles[x] for x in q['negative_features']]
    joined=pd.concat(columns,axis=1)
    return joined.min(axis=1).where(joined.notna().all(axis=1),-2.)


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    config_path=COHORT/'protein_concordance_v2/RNA_BASELINES.json'
    protocol_path=COHORT/'protein_concordance_v2/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    manifest=json.loads((COHORT/'protein_concordance_v2/protein_value_read_manifest.json').read_text())
    assert sha(config_path)==manifest['baseline_config_sha256']
    pool_path=COHORT/'rna_inputs_v1/mechanically_valid_candidates.csv'
    assert sha(pool_path)==protocol['candidate_pool_sha256']
    pool=pd.read_csv(pool_path);ids=pool.public_case_id.to_numpy(dtype=str)
    execution=pd.read_csv(INPUT/'execution_summary.csv')
    assert len(execution)==40 and execution['samples'].sum()==942*4 and execution.finite.eq(execution['samples']).all()
    compact_z=pd.read_csv(INPUT/'compact_zscore_scores.csv').set_index('sample_id').loc[ids]
    compact_rank=pd.read_csv(INPUT/'compact_rank_scores.csv').set_index('sample_id').loc[ids]
    compact_pct=compact_z.rank(method='average',pct=True)
    mcp=pd.concat([pd.read_csv(INPUT/f'{c}_MCPcounter_log2tpm_scores.csv') for c in sorted(pool.cohort.unique())]).set_index('sample_id').loc[ids]
    epic=pd.concat([pd.read_csv(INPUT/f'{c}_EPIC_TRef/cell_fractions.csv') for c in sorted(pool.cohort.unique())]).set_index('sample_id').loc[ids]
    mcp=mcp.drop(columns='cohort').rank(method='average',pct=True)
    epic_numeric=pd.DataFrame({'T_cell':epic.CD4_Tcells+epic.CD8_Tcells,'B_cell':epic.Bcells,'myeloid':epic.Macrophages,
                               'stromal':epic.CAFs,'angiogenesis':epic.Endothelial,
                               'immune':epic[['Bcells','CD4_Tcells','CD8_Tcells','NKcells','Macrophages']].sum(axis=1)}).rank(method='average',pct=True)
    mcp_features={'T_cell':mcp['T cells'],'B_cell':mcp['B lineage'],
                  'myeloid':mcp[['Monocytic lineage','Myeloid dendritic cells','Neutrophils']].max(axis=1),
                  'stromal':mcp['Fibroblasts'],'angiogenesis':mcp['Endothelial cells'],'cytotoxic':mcp['Cytotoxic lymphocytes']}
    combined={}
    for feature in compact_pct:
        components=[compact_pct[feature]]
        if feature in mcp_features:components.append(mcp_features[feature])
        if feature in epic_numeric:components.append(epic_numeric[feature])
        parts=pd.concat(components,axis=1)
        combined[feature]=parts.mean(axis=1).where(parts.notna().all(axis=1))
    combined=pd.DataFrame(combined)
    combined_pct=combined.rank(method='average',pct=True)
    combined.to_csv(OUT/'combined_source_rank_means.csv',index_label='sample_id')
    combined_pct.to_csv(OUT/'combined_feature_percentiles.csv',index_label='sample_id')
    definitions=json.loads((STUDY/'queries.json').read_text())['queries']
    variants=[{'query_id':q['query_id'],**v} for q in definitions for v in q['queries']]
    matrices={}
    for method,features in [('compact_marker_zscore',compact_pct),('compact_marker_rank',compact_rank.rank(method='average',pct=True)),('combined_conventional',combined_pct)]:
        matrices[method]=np.stack([query_score(features,q).to_numpy() for q in definitions for _ in q['queries']])
    texts=[]
    for sample,row in combined_pct.iterrows():
        fields=[]
        for feature,value in row.items():
            level='unavailable' if pd.isna(value) else 'high' if value>=.75 else 'low' if value<=.25 else 'intermediate'
            fields.append(f'{feature.replace("_"," ")} signal: {level}')
        texts.append('; '.join(fields)+'.')
    metadata=pd.read_csv(INPUT/'metadata_texts.csv').set_index('sample_id').loc[ids].metadata_text.tolist()
    text_frame=pd.DataFrame({'sample_id':ids,'metadata_text':metadata,'fixed_conventional_text':texts})
    text_frame.to_csv(OUT/'baseline_texts.csv',index=False)
    encoder_path=ROOT/'text_encoder_minilm'
    encoder_hashes=json.loads((ROOT/'paired_candidates/SDY2823/composition_pilot_v2/protocol.json').read_text())['encoder_sha256']
    for relative,expected in encoder_hashes.items():assert sha(encoder_path/relative)==expected
    contract={'source_config_sha256':sha(config_path),'protein_protocol_sha256':sha(protocol_path),'RNA_input_contract_sha256':sha(INPUT/'preprocessing_contract.json'),
              'deconvolution_execution_sha256':sha(INPUT/'execution_summary.csv'),'pool_sha256':sha(pool_path),'query_sha256':sha(STUDY/'queries.json'),
              'driver_sha256':sha(__file__),'encoder_sha256':encoder_hashes,
              'combined_rule':'Mean fixed source percentiles, then percentile-rank each combined feature for conjunctive query scoring and fixed text; no outcome fitting',
              'scope':'RNA-only baselines, except explicit source-metadata extra-input controls; only a focused fixed-panel comparator, not universal open-ended biology'}
    path=OUT/'contract.json'
    if path.exists():assert json.loads(path.read_text())==contract
    else:path.write_text(json.dumps(contract,indent=2)+'\n')
    query_texts=[v['text'] for v in variants]
    unique=sorted(set(query_texts+metadata+texts))
    model=SentenceTransformer(str(encoder_path),device='cpu')
    z=model.encode(unique,batch_size=32,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
    index={text:i for i,text in enumerate(unique)}
    q=z[[index[text] for text in query_texts]]
    matrices['metadata_minilm']=q @ z[[index[text] for text in metadata]].T
    matrices['fixed_conventional_text_minilm']=q @ z[[index[text] for text in texts]].T
    tfidf=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
    vocab=tfidf.fit_transform(metadata)
    matrices['metadata_tfidf']=(tfidf.transform(query_texts) @ vocab.T).toarray()
    with gzip.open(OUT/'rankings.csv.gz','wt',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['method','query_id','variant','sample_id','rank','score'])
        for method,scores in matrices.items():
            assert scores.shape==(90,942) and np.isfinite(scores).all()
            for qi,v in enumerate(variants):
                for rank,i in enumerate(np.lexsort((ids,-scores[qi])),1):writer.writerow([method,v['query_id'],v['variant'],ids[i],rank,float(scores[qi,i])])
    report={'candidate_cases':942,'methods':list(matrices),'queries':30,'variants':90,'ranked_rows':len(matrices)*90*942,
            'unique_metadata_texts':len(set(metadata)),'unique_fixed_texts':len(set(texts)),
            'abstention_query_sample_pairs':{name:int((score==-2).sum()) for name,score in matrices.items()},
            'EPIC_nonconverged':execution.groupby('method').nonconverged.sum(min_count=1).dropna().to_dict(),
            'performance_evaluated':False,'protein_qrels_read':False}
    (OUT/'execution_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
