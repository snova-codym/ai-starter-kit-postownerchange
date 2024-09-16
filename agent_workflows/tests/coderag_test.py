#!/usr/bin/env python3

import os
import sys
import unittest
import yaml
import logging
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))
logger.info(f'kit_dir: {kit_dir}')
logger.info(f'repo_dir: {repo_dir}')

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.vectordb.vector_db import VectorDb
from utils.model_wrappers.api_gateway import APIGateway 
from utils.agents.static_RAG_with_coding import CodeRAG

examples = []

CONFIG_PATH = os.path.join(kit_dir, "config.yaml")
PERSIST_DIRECTORY = os.path.join(kit_dir, "tests", "vectordata", "my-vector-db")
TEST_DATA_PATH = os.path.join(kit_dir, "tests", "data", "test")
logger.info(f'config path: {CONFIG_PATH}')
logger.info(f'persist directory: {PERSIST_DIRECTORY}')
logger.info(f'test data path: {TEST_DATA_PATH}')

def get_config_info(CONFIG_PATH: str) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config["api"]
        llm_info =  config["llm"]
        embedding_model_info = config["embedding_model"]
        retrieval_info = config["retrieval"]
        prompts = config["prompts"]
        
        return api_info, llm_info, embedding_model_info, retrieval_info, prompts

def load_embedding_model(embedding_model_info: dict) -> None:
        embeddings = APIGateway.load_embedding_model(
            type=embedding_model_info["type"],
            batch_size=embedding_model_info["batch_size"],
            coe=embedding_model_info["coe"],
            select_expert=embedding_model_info["select_expert"]
            ) 
        return embeddings  

class CodeRAGCase(unittest.TestCase):
    @classmethod
    def 