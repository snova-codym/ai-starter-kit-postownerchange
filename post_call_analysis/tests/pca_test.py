#!/usr/bin/env python3
"""
Post Call Analysis (PCA) Test Script

This script tests the functionality of the Post Call Analysis using unittest.
It loads transcripts and tests the analysis capabilities.

Usage:
    python tests/pca_test.py

Returns:
    0 if all tests pass, or a positive integer representing the number of failed tests.
"""

import os
import sys
import shutil
import time
import unittest
import logging
from typing import List, Dict, Any
import glob
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths and variables
current_dir = os.getcwd()
print(current_dir)
kit_dir = current_dir
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from post_call_analysis.src import analysis, plot, asr

audio_save_location=(os.path.join(kit_dir,"data/conversations/audio"))
transcript_save_location=(os.path.join(kit_dir,"data/conversations/transcription"))
transcription_path = os.path.join(transcript_save_location,'911_call.csv')
transcription=pd.read_csv(transcription_path)
facts_path = os.path.join(kit_dir, 'data/documents/facts')
procedures_path =  os.path.join(kit_dir, 'data/documents/example_procedures.txt')
facts_urls = []
classes = ["undefined", "emergency", "general information", "sales", "complains"]
entities = ["name", "address", "city", "phone number"]
sentiments = ["positive", "neutral" ,"negative"] 

def convert_to_dialogue_structure(transcription):
    dialogue = ''  
    for _, row in transcription.iterrows():
        speaker = str(row['speaker'])
        text = str(row['text'])
        dialogue += speaker + ': ' + text + '\n'   
    return dialogue

def main():

    dialogue = convert_to_dialogue_structure(transcription)
    print('Finished converting dialogue')
    conversation = analysis.load_conversation(dialogue, transcription_path)
    print('Finished loading conversation')
    conversation_chunks = analysis.get_chunks(conversation)
    print('Finished chunking')
    result=analysis.call_analysis_parallel(conversation_chunks, documents_path=facts_path, facts_urls=facts_urls, procedures_path=procedures_path, classes_list=classes, entities_list=entities, sentiment_list=sentiments)

    print("\nConversation summary:")
    print(result["summary"])
    assert result["summary"], "Summary is empty"

    print("\nClassification:")
    print(result["classification"][0])
    assert result["classification"][0] in classes, "Invalid classification value"

    print("\nSentiment analysis:")
    print(result["sentiment"])
    assert result["sentiment"] in sentiments, "Invalid sentiment value"

    print("\nNPS prediction:")
    print(result["nps_analysis"])
    assert result["nps_analysis"], "NPS analysis is empty"
    print("Predicted NPS: %d"%(result['nps_score']))
    assert isinstance(result["nps_score"], int),"NPS score is not an integer"

    print("\nFactual accuracy analysis:")
    print("correct: %s"%(result["factual_analysis"]['correct']))
    assert isinstance(result["factual_analysis"]["correct"], bool), "Factual analysis 'correct' is not a boolean"
    print("errors: %s"%(result["factual_analysis"]['errors']))

    print("\nProcedures analysis")
    print("correct: %s"%(result["procedural_analysis"]['correct']))
    assert isinstance(result["procedural_analysis"]["correct"], bool), "Procedural analysis 'correct' is not a boolean"
    print("errors: %s"%(result["procedural_analysis"]['errors']))

    print("\nExtracted entities")
    entities_items = result["entities"].items()
    print(entities_items)
    for key in result["entities"].keys():
        assert key in entities, f"Invalid entity key: {key}"

    print("\nCall quality assessment:")
    print(result["quality_score"])

if __name__ == "__main__":
    sys.exit(main())