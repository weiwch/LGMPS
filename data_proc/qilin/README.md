[HOMEPAGE](#) (Placeholder for Qilin Homepage)

## Data Descriptions:

File organization:

```
Qilin
  ├── search.csv
  ├── recommendation.csv
  ├── dqa.csv
  ├── item_features.csv
  └── item_semantic_ids.csv
```

#### 1. Descriptions of the fields in search.csv

| Field Name | Description | Type |
| --- | --- | --- |
| search_id | The ID of the search request. | string |
| session_id | The ID of the user's search session. | string |
| user_id | The ID of the user. | string |
| query | The text of the user's search query. | string |
| query_from_type | The source or type of the query. | float64 |
| recent_clicked_note_ids | List of recently clicked note string IDs by the user. | string (list) |
| recent_clicked_note_idxs | List of recently clicked note integer index IDs. | string (list) |
| search_result_details | JSON string detailing the items returned, including their original string IDs and user interactions. | string (JSON) |
| search_result_details_with_idx | JSON string detailing the items returned, using integer `note_idx` for easier embedding lookup. | string (JSON) |

**Details of `search_result_details_with_idx` JSON Array:**
*   `note_idx`: The integer ID of the viewed item.
*   `click`: Whether the user clicks the item (1 for clicked, 0 for not).
*   `page_time`: The duration the item was viewed.
*   `position`: The ranking position of the item.
*   `search_timestamp`: Unix timestamp of the search.
*   `collect`, `comment`, `like`, `share`: Other implicit user feedback signals.

#### 2. Descriptions of the fields in recommendation.csv

| Field Name | Description | Type |
| --- | --- | --- |
| request_id | The ID of the recommendation request. | string |
| session_id | The ID of the recommendation session. | string |
| user_id | The ID of the user. | string |
| recent_clicked_note_ids | List of recently clicked note string IDs. | string (list) |
| recent_clicked_note_idxs | List of recently clicked note integer index IDs. | string (list) |
| rec_result_details | JSON string detailing the recommended items with original string IDs. | string (JSON) |
| rec_result_details_with_idx | JSON string detailing the recommended items with integer `note_idx`. | string (JSON) |

Similar to the search logs, the recommendation details include user interaction signals like `click` and `page_time`.

#### 3. Descriptions of the fields in dqa.csv (Direct Question Answering)

| Field Name | Description | Type |
| --- | --- | --- |
| search_id | The ID of the search request. | string |
| session_id | The ID of the session. | string |
| user_id | The ID of the user. | string |
| query | The text of the user's search query. | string |
| dqa_output | Output related to the Direct Question Answering system. | float64 |
| ref_note_id_list | List of reference note string IDs used for DQA. | string (list) |
| ref_note_idx_list | List of reference note integer indices used for DQA. | string (list) |
| is_like_clk | DQA specific interaction flag. | float64 |
| is_onebox_trace_clk | DQA specific interaction flag. | float64 |
| is_content_clk | DQA specific interaction flag. | float64 |
| is_experience_clk | DQA specific interaction flag. | float64 |
| search_result_details_with_idx | JSON string of the returned items. | string (JSON) |

#### 4. Descriptions of the fields in item_features.csv

| Field Name | Description | Type |
| --- | --- | --- |
| note_idx | The integer ID of the item/note (used to map across tables). | int64 |
| text_for_embedding | A concatenated string of the note's Title, Content, and Categories, ready for LLM processing. | string |
| first_image_path | The relative path to the primary image for the note. | string |

## Analysis

The Qilin dataset captures rich user interactions across distinct scenarios: **Search**, **Recommendation**, and **Direct Question Answering (DQA)**.

Unlike datasets that only capture singular item interactions per row, the Qilin dataset groups all returned items for a specific request into a JSON array (`_details_with_idx`). This structure perfectly preserves the **ranking context** and the **presentation order** (`position`) of items as they were shown to the user.

Users generate implicit feedback on these lists, primarily tracked via the `click` metric. By parsing the `text_for_embedding`, models can directly leverage multi-modal or text-based embeddings corresponding to the exact items presented in the sequence.
