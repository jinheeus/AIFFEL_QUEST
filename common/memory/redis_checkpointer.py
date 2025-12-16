import pickle
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from redis import Redis


class RedisSaver(BaseCheckpointSaver):
    """A checkpoint saver that stores checkpoints in a Redis database."""

    def __init__(self, client: Redis, ttl: int = 86400):
        super().__init__()
        self.client = client
        self.ttl = ttl

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the Redis database based on the
        provided config. If the config contains a "checkpoint_id", the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest
        checkpoint for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no
            matching checkpoint is found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if checkpoint_id:
            key = f"checkpoint:{thread_id}:{checkpoint_id}"
            data = self.client.get(key)
        else:
            # Get latest
            # We store the latest checkpoint ID in a separate key or use a sorted set.
            # Simple approach: We track the "latest" ID for a thread.
            latest_key = f"checkpoint_latest:{thread_id}"
            latest_id = self.client.get(latest_key)
            if not latest_id:
                return None
            key = f"checkpoint:{thread_id}:{latest_id.decode()}"
            data = self.client.get(key)

            # Refresh TTL for latest pointer on read
            if latest_id:
                self.client.expire(latest_key, self.ttl)
                self.client.expire(key, self.ttl)

        if not data:
            return None

        # Deserialize
        checkpoint, metadata, parent_config = pickle.loads(data)

        # Construct config for this checkpoint
        checkpoint_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint["id"],
            }
        }

        return CheckpointTuple(
            config=checkpoint_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
        )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async version of get_tuple."""
        import asyncio

        return await asyncio.to_thread(self.get_tuple, config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        """Async version of put."""
        import asyncio

        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        For simplicity in this PoC, we might just return empty or recent.
        Full implementation requires Sorted Sets in Redis (ZSET).
        """
        # Placeholder for full history listing
        # If we need history, we should implement ZRANGE logic.
        # For now, we focus on get/put for basic persistence.
        yield from []

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the Redis database. The checkpoint is associated
        with the provided config and strictly monotonically increasing checkpoint ID.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): The metadata to associate with the checkpoint.
            new_versions (Dict[str, Any]): New versions of the state keys.

        Returns:
            RunnableConfig: The updated config containing the saved checkpoint ID.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]

        # Filter config to remove unpicklable objects (like callbacks, streams)
        # We only strictly need 'configurable' to restore context.
        # STRIOT FILTERING: Only keep known safe keys.
        safe_keys = {"thread_id", "checkpoint_ns", "checkpoint_id"}
        safe_configurable = {
            k: v for k, v in config.get("configurable", {}).items() if k in safe_keys
        }
        saved_config = {"configurable": safe_configurable}

        # Serialize everything using the sanitized config
        try:
            data = pickle.dumps((checkpoint, metadata, saved_config))
        except TypeError as e:
            print(f"❌ Pickle Error: {e}")
            # Fallback: Try saving without config if it still fails (though unlikely)
            # Or inspect checkpoint for issues.
            # For now, just re-raise with context
            raise e

        # Save exact checkpoint
        key = f"checkpoint:{thread_id}:{checkpoint_id}"
        self.client.set(key, data, ex=self.ttl)

        # Update latest pointer
        latest_key = f"checkpoint_latest:{thread_id}"
        self.client.set(latest_key, checkpoint_id, ex=self.ttl)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Save intermediate writes to the database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        # Serialize writes
        data = pickle.dumps(writes)

        # Key: writes:thread_id:checkpoint_id:task_id
        key = f"writes:{thread_id}:{checkpoint_id}:{task_id}"
        self.client.set(key, data, ex=self.ttl)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Async version of put_writes."""
        import asyncio

        return await asyncio.to_thread(self.put_writes, config, writes, task_id)
