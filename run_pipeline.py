import sys
import os
import pandas as pd
from tqdm import tqdm
# Setup LangFuse logging
from dotenv import load_dotenv
load_dotenv(dotenv_path='./notebooks/.env')
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

langfuse = Langfuse(
  secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
  public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
  host=os.getenv('LANGFUSE_HOST')
)

# Import all agents and vector store functions
sys.path.append(os.path.abspath('./src'))
from agents import run_jd_extraction_agent, run_resume_extraction_agent, run_evaluator_agent
from vector_store import get_vector_store, retrieve_context

def format_list_for_excel(item_list):
    """Converts a Python list into a readable bulleted string for spreadsheet cells."""
    if not item_list:
        return "None identified."
    return "- " + "\n- ".join(item_list)

def calculate_performance_metrics(df_results):
    """
    Calculates the Spearman Rank Correlation and tier averages.
    
    """
    # Maps text tiers to numerical ranks for correlation math
    tier_weights = {
        "Strong Match": 3,
        "Average Match": 2,
        "Bad Match": 1
    }
    
    # Create a numerical column for the math
    df_results['Tier_Numeric'] = df_results['Ground_Truth_Tier'].map(tier_weights)
    
    # Calculate Spearman Rank Correlation using native pandas
    # We use min_periods=3 to ensure it doesn't crash on tiny datasets
    spearman_corr = df_results['Match_Score'].corr(df_results['Tier_Numeric'], method='spearman', min_periods=3)
    
    # Calculate Average & Median Score per Tier
    tier_summary = df_results.groupby('Ground_Truth_Tier')['Match_Score'].agg(['mean', 'median', 'count']).reset_index()
    tier_summary.rename(columns={'mean': 'Average_Match_Score', 'median': 'Median_Match_Score', 'count': 'Sample_Size'}, inplace=True)
    
    return spearman_corr, tier_summary

def main():
    input_file = './datasets/jd_resume_pairs.csv'
    output_file = './datasets/final_evaluation_report.xlsx'
    
    print(f"Loading candidate dataset and vector store...")
    df = pd.read_csv(input_file)
    v_store = get_vector_store()
    
    unique_jds = df[['source_jd_title', 'source_jd_description']].drop_duplicates()
    evaluation_records = []
    
    print(f"Found {len(unique_jds)} unique roles. Initiating 2-Step RAG Pipeline...\n")
    
    for jd_index, jd_row in tqdm(unique_jds.iterrows(), total=len(unique_jds), desc="Processing Roles"):
        target_role_title = jd_row['source_jd_title']
        raw_jd = jd_row['source_jd_description']
        
        try:
            # STAGE 1: Vector Search
            jd_json = run_jd_extraction_agent(raw_jd)
            dense_query = jd_json.get("dense_search_string", raw_jd[:300]) 
            
            raw_chunks = retrieve_context(dense_query, v_store, strategy="hybrid", top_n=60)
            
            top_10_cv_ids = []
            for chunk in raw_chunks:
                cv_id = chunk.metadata.get("cv_id")
                if cv_id is not None and cv_id not in top_10_cv_ids:
                    top_10_cv_ids.append(cv_id)
                if len(top_10_cv_ids) == 10:
                    break
                    
            # STAGE 2: Json Match
            for rank, candidate_id in enumerate(top_10_cv_ids):
                raw_cv = df.loc[candidate_id, 'generated_resume_text']
                # Find out which role this CV was originally generated for
                cv_native_role = df.loc[candidate_id, 'source_jd_title']
                actual_tier = df.loc[candidate_id, 'assigned_tier']
                if cv_native_role != target_role_title:
                    # If retrieved for the wrong job, they are objectively a Bad Match for THIS campaign
                    actual_tier = "Bad Match" 
                else:
                    # If it's the right job, keep their intended ground truth
                    actual_tier = df.loc[candidate_id, 'assigned_tier']
                # Create a dynamic config for this specific candidate evaluation
                trace_config = {
                    "callbacks": [langfuse_handler],
                    "run_name": f"Eval: {target_role_title}",
                    "tags": [f"CV_ID:{candidate_id}", actual_tier],
                    "metadata": {
                    "langfuse_session_id": "pipeline_evaluation_v3"

                    }
                }

                cv_json = run_resume_extraction_agent(raw_cv, config=trace_config)
                eval_result = run_evaluator_agent(jd_json, cv_json, config=trace_config)
                
                flat_record = {
                    "Target_Role": target_role_title,
                    "Rank_in_Search": rank + 1,
                    "CV_ID": candidate_id,
                    "Ground_Truth_Tier": actual_tier,
                    "Recommendation": eval_result.get("recommendation", "Error"),
                    "Match_Score": eval_result.get("match_score", 0),
                    "Justification": eval_result.get("justification", ""),
                    "Strengths": format_list_for_excel(eval_result.get("strengths", [])),
                    "Critical_Gaps": format_list_for_excel(eval_result.get("critical_gaps", []))
                }
                
                evaluation_records.append(flat_record)
                
        except Exception as e:
            print(f"\nError processing Role '{target_role_title}': {e}")
            evaluation_records.append({
                "Target_Role": target_role_title,
                "Recommendation": "PIPELINE_ERROR",
                "Justification": str(e)
            })

   
    # EXPORT RESULTS & CALCULATE METRICS
   
    print("\nBatch processing complete. Calculating performance metrics...")
    df_results = pd.DataFrame(evaluation_records)
    
    if not df_results.empty and "Rank_in_Search" in df_results.columns:
        df_results = df_results.sort_values(by=["Target_Role", "Rank_in_Search"])
        
        # Calculate metrics
        spearman_corr, tier_summary = calculate_performance_metrics(df_results)
        
        # Print summary to terminal
        print("\n=== PIPELINE PERFORMANCE SUMMARY ===")
        print(f"Spearman Rank Correlation (0 to 1): {spearman_corr:.3f}")
        print("\nScore Distribution by Tier:")
        print(tier_summary.to_markdown(index=False))
        
        # Save to Excel with Multiple Sheets
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Save the evaluation scores for the top candidates for each job
            df_results.drop(columns=['Tier_Numeric'], errors='ignore').to_excel(writer, sheet_name='Candidate_Evaluations', index=False)
            
            # Sheet 2: The aggregated summary metrics
            tier_summary.to_excel(writer, sheet_name='Performance_Summary', index=False)
            
            # Add the Spearman score to Sheet 2
            pd.DataFrame([{"Metric": "Spearman Rank Correlation", "Value": spearman_corr}]).to_excel(
                writer, sheet_name='Performance_Summary', startrow=len(tier_summary) + 2, index=False
            )
            
    print(f"\nSuccess! Detailed report and summary metrics saved to {output_file}")

if __name__ == "__main__":
    main()