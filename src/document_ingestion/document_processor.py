from typing import List, Union, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import(
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    PyPDFDirectoryLoader
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Check if pymupdf is available for image extraction
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. PDF image extraction will be disabled. Install with: pip install pymupdf")

class DocumentProcessor:
    """Handle document loading and processing with image extraction support."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, extract_images: bool = True):
        """
        Initialize DocumentProcessor with text splitting configuration.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            extract_images: Whether to extract and describe images from PDFs (requires pymupdf)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images and PYMUPDF_AVAILABLE
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, extract_images={self.extract_images}")
    
    def load_from_url(self, urls: List[str]) -> List[Document]:
        """
        Load documents from URLs.

        Args:
            urls: List of URLs to load

        Returns:
            List of loaded documents
        """
        logger.info(f"Loading {len(urls)} URLs")
        docs = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
                logger.info(f"Successfully loaded URL: {url}")
            except Exception as e:
                logger.error(f"Failed to load URL {url}: {str(e)}")
        return docs 
    
    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a PDF file with optional image extraction.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of loaded documents
        """
        try:
            logger.info(f"Loading PDF: {file_path}")

            if self.extract_images and PYMUPDF_AVAILABLE:
                docs = self._load_pdf_with_images(file_path)
            else:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()

            logger.info(f"Successfully loaded {len(docs)} pages from {file_path}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {str(e)}")
            return []

    def _load_pdf_with_images(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load PDF with text and image extraction using PyMuPDF.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of documents with text and image descriptions
        """
        docs = []
        file_path = Path(file_path)

        try:
            pdf_document = fitz.open(str(file_path))

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Extract text
                text_content = page.get_text()

                # Extract images
                image_descriptions = self._extract_images_from_page(page, page_num, file_path.name)

                # Combine text and image descriptions
                full_content = text_content
                if image_descriptions:
                    full_content += "\n\n--- Images on this page ---\n"
                    full_content += "\n".join(image_descriptions)

                # Create document for this page
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "source": str(file_path),
                        "page": page_num,
                        "total_pages": len(pdf_document),
                        "has_images": len(image_descriptions) > 0,
                        "image_count": len(image_descriptions)
                    }
                )
                docs.append(doc)

            pdf_document.close()
            return docs

        except Exception as e:
            logger.error(f"Error loading PDF with images: {str(e)}")
            # Fallback to standard loader
            loader = PyPDFLoader(str(file_path))
            return loader.load()

    def _extract_images_from_page(self, page, page_num: int, filename: str) -> List[str]:
        """
        Extract images from a PDF page and generate descriptions.

        Args:
            page: PyMuPDF page object
            page_num: Page number
            filename: Name of the PDF file

        Returns:
            List of image description strings
        """
        image_descriptions = []

        try:
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)

                    if base_image:
                        image_ext = base_image.get("ext", "unknown")
                        image_width = base_image.get("width", 0)
                        image_height = base_image.get("height", 0)
                        image_size = len(base_image.get("image", b""))

                        description = (
                            f"[Image {img_index + 1}: {image_ext.upper()} format, "
                            f"{image_width}x{image_height} pixels, "
                            f"{image_size / 1024:.1f} KB]"
                        )
                        image_descriptions.append(description)

                except Exception as img_e:
                    logger.debug(f"Could not extract image {img_index} from page {page_num}: {str(img_e)}")

        except Exception as e:
            logger.debug(f"Error extracting images from page {page_num}: {str(e)}")

        return image_descriptions
    
    def load_from_directory(self, directory_path: Union[str, Path]) -> List[Document]:
        """
        Load all PDF documents from a directory with image extraction support.

        Args:
            directory_path: Path to the directory

        Returns:
            List of loaded documents
        """
        try:
            logger.info(f"Loading PDFs from directory: {directory_path}")
            directory_path = Path(directory_path)
            docs = []

            if self.extract_images and PYMUPDF_AVAILABLE:
                # Load each PDF individually to support image extraction
                pdf_files = list(directory_path.glob("*.pdf")) + list(directory_path.glob("*.PDF"))
                for pdf_file in pdf_files:
                    docs.extend(self.load_from_pdf(pdf_file))
            else:
                loader = PyPDFDirectoryLoader(str(directory_path))
                docs = loader.load()

            logger.info(f"Successfully loaded {len(docs)} documents from {directory_path}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load directory {directory_path}: {str(e)}")
            return []    
    
    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a text file.

        Args:
            file_path: Path to the text file

        Returns:
            List of loaded documents
        """
        try:
            logger.info(f"Loading text file: {file_path}")
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs = loader.load()
            logger.info(f"Successfully loaded text file {file_path}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {str(e)}")
            return []

    def load_documents(self, sources: List[Union[str, Path]]) -> List[Document]:
        """
        Load documents from URLs, PDFs, directories, or txt files.

        Args:
            sources: List of sources (URLs, file paths, or directory paths)

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for src in sources:
            src_str = str(src)
            if src_str.startswith("http://") or src_str.startswith("https://"):
                docs.extend(self.load_from_url([src_str]))
            else:
                path = Path(src)
                if path.is_dir():
                    docs.extend(self.load_from_directory(path))
                elif path.suffix.lower() == ".txt":
                    docs.extend(self.load_from_txt(path))
                elif path.suffix.lower() == ".pdf":
                    docs.extend(self.load_from_pdf(path))
                else:
                    logger.warning(f"Unsupported file type: {src}. Skipping.")
        logger.info(f"Total documents loaded: {len(docs)}")
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        logger.info(f"Splitting {len(documents)} documents")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def process_url(self, url: str) -> List[Document]:
        """
        Load and split documents from a URL.

        Args:
            url: URL to process

        Returns:
            List of processed document chunks
        """
        docs = self.load_documents([url])
        return self.split_documents(docs)



