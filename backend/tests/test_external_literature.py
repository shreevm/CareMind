from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.caremind.config import Settings
from backend.caremind.external_literature import ExternalLiteratureSearch


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "CAREMIND_SQLITE_PATH": tmp_path / "caremind.db",
        "CAREMIND_DATA_DIR": tmp_path,
        "CAREMIND_UPLOAD_DIR": tmp_path / "uploads",
        "supabase_url": None,
        "supabase_secret_key": None,
        "PINECONE_API_KEY": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_pubmed_query_sanitizes_obvious_patient_names(tmp_path: Path) -> None:
    search = ExternalLiteratureSearch(make_settings(tmp_path))

    safe_query = search.sanitize_query("Find a similar case report for Mr Venkat Ramanujam with pneumonia")

    assert "Venkat" not in safe_query
    assert "Ramanujam" not in safe_query
    assert "pneumonia" in safe_query.lower()
    assert "case report" in safe_query.lower()


def test_pubmed_xml_parsing_returns_retrieved_chunks(tmp_path: Path) -> None:
    search = ExternalLiteratureSearch(make_settings(tmp_path))
    xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <ArticleTitle>Example pneumonia case report</ArticleTitle>
            <Journal><Title>Example Journal</Title></Journal>
            <Abstract><AbstractText>A patient presented with cough and fever.</AbstractText></Abstract>
            <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    chunks = search._parse_pubmed_xml(xml)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "pubmed:12345"
    assert "Example pneumonia case report" in chunks[0].document_name
    assert "cough and fever" in chunks[0].text


def test_medlineplus_query_sanitizes_patient_names_without_case_report(tmp_path: Path) -> None:
    search = ExternalLiteratureSearch(make_settings(tmp_path))

    safe_query = search.sanitize_health_topic_query("Explain pneumonia for Mr Venkat Ramanujam")

    assert "Venkat" not in safe_query
    assert "Ramanujam" not in safe_query
    assert "pneumonia" in safe_query.lower()
    assert "case report" not in safe_query.lower()


def test_medlineplus_xml_parsing_returns_health_topic_chunks(tmp_path: Path) -> None:
    search = ExternalLiteratureSearch(make_settings(tmp_path))
    xml = """
    <nlmSearchResult>
      <list>
        <document rank="1">
          <content name="title">Pneumonia</content>
          <content name="url">https://medlineplus.gov/pneumonia.html</content>
          <content name="groupName">Lungs and Breathing</content>
          <content name="FullSummary">&lt;p&gt;Pneumonia is an infection in one or both lungs.&lt;/p&gt;</content>
        </document>
      </list>
    </nlmSearchResult>
    """

    chunks = search._parse_medlineplus_xml(xml)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "medlineplus:pneumonia"
    assert chunks[0].document_name == "MedlinePlus: Pneumonia"
    assert "Pneumonia is an infection" in chunks[0].text
    assert chunks[0].metadata["source"] == "medlineplus"
