import os
import json
import logging
import networkx as nx
from google.cloud import storage
from google.api_core import exceptions as google_api_exceptions
from google.cloud import exceptions as google_cloud_exceptions
from datetime import datetime, timedelta

# Configure logging (integrates with Cloud Logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration from Environment Variables ---
# Set these during Cloud Function deployment
BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'your-default-bucket-name') # REQUIRED
BLOB_NAME = os.environ.get('GCS_BLOB_NAME', 'graph_data.json')
STALE_THRESHOLD_HOURS = int(os.environ.get('STALE_THRESHOLD_HOURS', '24'))

# Initialize GCS client globally (recommended practice in Cloud Functions)
# Client initialization is relatively fast after the first invocation (warm instance)
storage_client = storage.Client()

def prune_graph_cloud_function(event, context):
    """
    Google Cloud Function to prune stale nodes from the object graph in GCS.
    Triggered by Cloud Scheduler (e.g., via Pub/Sub) or GCS event.

    Args:
        event (dict): Event payload (e.g., Pub/Sub message data or GCS object metadata).
                      Not directly used in this version but available.
        context (google.cloud.functions.Context): Metadata about the event.
    """
    logging.info(f"Function triggered by event ID: {context.event_id}, type: {context.event_type}")
    logging.info(f"Using Bucket: {BUCKET_NAME}, Blob: {BLOB_NAME}, Threshold: {STALE_THRESHOLD_HOURS} hours")

    if not BUCKET_NAME or BUCKET_NAME == 'your-default-bucket-name':
        logging.error("GCS_BUCKET_NAME environment variable not set or using default.")
        return  # Or raise error

    graph = None
    blob = None

    try:
        # 1. Access graph data
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)

        logging.info(f"Downloading graph data from gs://{BUCKET_NAME}/{BLOB_NAME}...")
        try:
            graph_json_string = blob.download_as_string()
        except google_cloud_exceptions.NotFound:
            logging.warning(f"Blob gs://{BUCKET_NAME}/{BLOB_NAME} not found. Assuming empty graph or first run.")
            # Optional: Create an empty graph if it doesn't exist? Or just exit?
            # For now, exit gracefully if file doesn't exist.
            return

        logging.info("Parsing graph JSON...")
        graph_data = json.loads(graph_json_string)
        graph = nx.node_link_graph(graph_data)
        logging.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

        # 2. Identify stale nodes
        threshold = timedelta(hours=STALE_THRESHOLD_HOURS)
        now = datetime.utcnow() # Use UTC for consistency
        stale_nodes = []
        nodes_processed = 0
        nodes_missing_ts = 0

        for node_id, node_data in graph.nodes(data=True):
            nodes_processed += 1
            timestamp_str = node_data.get('timestamp')
            if timestamp_str:
                try:
                    # Ensure timestamp is timezone-aware if needed, or make `now` offset-naive
                    # Assuming timestamps are UTC from isoformat()
                    node_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')) # Handle 'Z' if present
                    # Make comparison timezone-naive if necessary based on how 'now' is defined
                    node_time_naive = node_time.replace(tzinfo=None) # Example if now is naive
                    if now - node_time_naive > threshold:
                        stale_nodes.append(node_id)
                except ValueError:
                    logging.warning(f"Could not parse timestamp '{timestamp_str}' for node {node_id}. Skipping staleness check.")
                except TypeError:
                     logging.warning(f"Timestamp for node {node_id} is not a string ('{timestamp_str}'). Skipping staleness check.")
            else:
                nodes_missing_ts += 1
                logging.debug(f"Node {node_id} missing 'timestamp' attribute. Skipping staleness check.")

        if nodes_missing_ts > 0:
            logging.warning(f"{nodes_missing_ts}/{nodes_processed} nodes were missing timestamp attribute.")

        logging.info(f"Identified {len(stale_nodes)} stale nodes to prune.")

        # 3. Delete stale nodes (and their edges implicitly)
        if stale_nodes:
            graph.remove_nodes_from(stale_nodes)
            logging.info(f"Removed stale nodes. New graph size: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

            # 4. Update the graph data in GCS
            logging.info(f"Uploading pruned graph back to gs://{BUCKET_NAME}/{BLOB_NAME}...")
            updated_graph_data = nx.node_link_data(graph)
            updated_graph_json = json.dumps(updated_graph_data, indent=2) # Add indent for readability if desired
            blob.upload_from_string(updated_graph_json, content_type='application/json')
            logging.info("Graph pruned and updated successfully!")
        else:
            logging.info("No stale nodes found. No update needed.")

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from blob gs://{BUCKET_NAME}/{BLOB_NAME}: {e}", exc_info=True)
    except google_cloud_exceptions.Forbidden as e:
        logging.error(f"Permission denied accessing GCS bucket '{BUCKET_NAME}' or blob '{BLOB_NAME}': {e}", exc_info=True)
    except google_api_exceptions.GoogleAPIError as e:
        logging.error(f"Google Cloud API error during GCS operation: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
