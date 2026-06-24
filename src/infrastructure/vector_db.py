import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from supabase.client import create_client, Client

from src.config import settings, PROJECT_ROOT

logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# 🛠️ Workaround for Bug #410 LangChain (Supabase update)
# -------------------------------------------------------------
class FixedSupabaseVectorStore(SupabaseVectorStore):
    """
    Overrides the default SupabaseVectorStore to fix the similarity_search_with_score method.
    Temporary workaround for LangChain bug #410.
    """
    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        
        embedder = getattr(self, "embeddings", None) or getattr(self, "_embedding", None) or getattr(self, "embedding", None)
        client = getattr(self, "_client", None) or getattr(self, "client", None)
        q_name = getattr(self, "_query_name", None) or getattr(self, "query_name", "match_documents")
        
        if not embedder or not client:
            raise ValueError("LangChain internal attributes changed. Check FixedSupabaseVectorStore.")
        
        embedding_vector = embedder.embed_query(query)

        match_documents_params = {
            "query_embedding": embedding_vector,
            "match_count": k,
            "filter": filter or {},
        }
        
        res = client.rpc(q_name, match_documents_params).execute()

        return [
            (
                Document(
                    metadata=search.get("metadata", {}),
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content", "")
        ]
# -------------------------------------------------------------

class VectorStoreManager:
    """Manages embeddings and vector-store collections for the RAG pipeline.

    Supports two backends, selected by ``settings.vector_db.provider``:

    - **Chroma** (local): stores vectors on disk under the configured data dir.
    - **Supabase** (cloud): stores vectors in a Supabase PostgreSQL table via pgvector.

    Embeddings are generated with a HuggingFace sentence-transformer model and
    are lazily initialised on first use to avoid startup latency.
    """
    def __init__(self):
        # Multi-Collection Config
        directories_config = settings.get("directories", {})
        self.collections_config = settings.get("collections", [
            {
                "name": settings.get("vector_db.collection_name", "default_collection"),
                "data_dir": directories_config.get("data_dir", "data")
            }
        ])

        self.embedding_model_name: str = settings.get("models.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size: int = settings.get("models.chunk_size", 1000)
        self.chunk_overlap: int = settings.get("models.chunk_overlap", 200)

        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._supabase: Optional[Client] = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Loads the HuggingFace embedding model lazily."""
        if self._embeddings is None:
            logger.debug(f"Loading embedding model: {self.embedding_model_name}")
            self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        return self._embeddings
    
    @property
    def supabase(self) -> Client:
        """Initializes the Supabase Client lazily."""
        if self._supabase is None:
            supabase_url = settings.database.get("supabase_url")
            supabase_key = settings.database.get("supabase_key")

            if not supabase_url or not supabase_key:
                raise ValueError("Missing Supabase credentials in environment variables.")

            self._supabase = create_client(supabase_url, supabase_key)
            logger.info("✅ Connected to Supabase successfully.")
        return self._supabase

    def _enrich_metadata(self, chunk: Document, collection_name: str) -> None:
        """Enriches chunk metadata dynamically based on folder structure and settings.yaml."""
        chunk.metadata["collection"] = collection_name

        source_path = os.path.normpath(chunk.metadata.get("source", "")).lower()
        content_lower = chunk.page_content.lower()

        chunk.metadata["doc_type"] = "directory" if "2_preferred_vendors" in source_path else "policy"

        folder_mapping = settings.get("ingestion_rules.folder_mapping", {
            "1_te_policies_compliance": "compliance_rules",
            "2_preferred_vendors": "inventory_catalogs",
            "3_risk_and_duty_of_care": "security_protocols",
            "4_expense_workflows": "admin_workflows",
        })
        chunk.metadata["category"] = "general"
        for folder_name, category_tag in folder_mapping.items():
            if folder_name in source_path:
                chunk.metadata["category"] = category_tag
                break

        file_specific_tags = settings.get("ingestion_rules.file_specific_tags", {
            "per_diem_limits": "financial_caps",
            "sustainability_transport": "transport_rules",
            "executive_tier": "executive_privileges",
            "hotel_partners_directory": "inventory",
        })
        chunk.metadata["policy_type"] = "general"
        for file_key, tag in file_specific_tags.items():
            if file_key in source_path:
                chunk.metadata["policy_type"] = tag
                break

        target_cities = settings.get(
            "ingestion_rules.target_cities",
            ["madrid", "london", "tokyo", "new york", "valencia", "barcelona"],
        )
        for city in target_cities:
            if re.search(r"\b" + re.escape(city) + r"\b", content_lower):
                chunk.metadata["city"] = city
                break

    def _get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Helper method to ensure consistent chunking across all ingestions."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            add_start_index=True 
        )

    # =========================================================================
    # 1: BATCH INGESTION (SUPABASE CLOUD)
    # =========================================================================
    def ingest_documents(self) -> None:
        for collection_info in self.collections_config:
            collection_name = collection_info.get("name")
            abs_data_dir = str(PROJECT_ROOT / collection_info.get("data_dir"))

            logger.info(f"\n--- Processing Collection for Supabase: {collection_name} ---")

            if not os.path.exists(abs_data_dir):
                logger.warning(f"Directory not found: '{abs_data_dir}'. Skipping.")
                continue

            loaders = [
                DirectoryLoader(abs_data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True),
                DirectoryLoader(abs_data_dir, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True),
                DirectoryLoader(abs_data_dir, glob="**/*.docx", loader_cls=Docx2txtLoader, silent_errors=True)
            ]
            
            documents: List[Document] = []
            for loader in loaders:
                documents.extend(loader.load())
                
            if not documents:
                logger.warning(f"No documents found in {abs_data_dir}. Skipping...")
                continue

            chunks: List[Document] = self._get_text_splitter().split_documents(documents)
            if not chunks:
                continue
                
            for chunk in chunks:
                self._enrich_metadata(chunk, collection_name)

            logger.info(f"Uploading {len(chunks)} chunks with metadata to Supabase...")
            
            FixedSupabaseVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.supabase,
                table_name="documents",
                query_name="match_documents"
            )
            logger.info(f"✅ Ingestion complete for '{collection_name}'.")

    # =========================================================================
    # 2: WEB APP INGESTION (In-Memory ChromaDB)
    # =========================================================================
    def ingest_single_document(self, file_path: str, collection_name: str = "temp_chat") -> VectorStoreRetriever:
        logger.info(f"Ingesting temp file for chat: {file_path}")
        
        file_ext = file_path.lower().split('.')[-1]
        loader_map = {
            'pdf': PyPDFLoader,
            'txt': TextLoader,
            'docx': Docx2txtLoader
        }
        
        if file_ext not in loader_map:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        loader = loader_map[file_ext](file_path)
        
        # Usamos el splitter refactorizado
        chunks = self._get_text_splitter().split_documents(loader.load())

        db = Chroma(
            collection_name=collection_name,
            persist_directory=None,  
            embedding_function=self.embeddings
        )
        db.add_documents(documents=chunks)
        
        return db.as_retriever(search_kwargs={"k": settings.get("retrieval.top_k", 4)})

    # =========================================================================
    # 3: RETRIEVAL (SUPABASE)
    # =========================================================================
    def get_retriever(self, collection_name: str, metadata_filter: Optional[Dict[str, Any]] = None) -> VectorStoreRetriever:
        vector_store = FixedSupabaseVectorStore(
            client=self.supabase,
            embedding=self.embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        final_filter = metadata_filter or {}
        final_filter["collection"] = collection_name
        
        search_kwargs = {
            "k": settings.get("retrieval.top_k", 4),
            "filter": final_filter
        }
        
        return vector_store.as_retriever(search_kwargs=search_kwargs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    manager = VectorStoreManager()
    manager.ingest_documents()