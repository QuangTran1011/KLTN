from pyspark.sql import functions as F, Window
from loguru import logger

class SparkRichnessDataSampler:
    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "item_id",
        ts_col: str = "timestamp",
        random_seed: int = 41,
        min_user_interactions: int = 5,
        min_item_interactions: int = 10,
        debug: bool = False,
        num_partitions: int = 128  
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.ts_col = ts_col
        self.random_seed = random_seed
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.debug = debug
        self.num_partitions = num_partitions

    def sample(self, df, train_ratio=0.8):
        keep_sampling = True
        iteration = 0
        max_iterations = 20
        
        # checkpoint để truncate lineage
        df = df.checkpoint(eager=True)
        persisted_dfs = []  # Track để unpersist
        
        while iteration < max_iterations and keep_sampling:
            keep_sampling = False
            if self.debug:
                logger.info(f"Begin iteration {iteration}")
            
            # Repartition theo user trước join để tránh skew
            df = df.repartition(self.num_partitions, self.user_col)
            
            # Loại user không đủ điều kiện
            uu = (df.groupBy(self.user_col)
                    .count()
                    .filter(F.col("count") < self.min_user_interactions)
                    .select(self.user_col))
            
            if not uu.rdd.isEmpty():
                keep_sampling = True
                old_df = df
                df = df.join(uu, on=self.user_col, how="left_anti").persist()
                persisted_dfs.append((old_df, df))
            
            # Repartition theo item trước join để tránh skew
            df = df.repartition(self.num_partitions, self.item_col)
            
            # Loại item không đủ điều kiện
            ui = (df.groupBy(self.item_col)
                    .count()
                    .filter(F.col("count") < self.min_item_interactions)
                    .select(self.item_col))
            
            if not ui.rdd.isEmpty():
                keep_sampling = True
                old_df = df
                df = df.join(ui, on=self.item_col, how="left_anti").persist()
                persisted_dfs.append((old_df, df))
            
            if self.debug:
                count_approx = df.rdd.countApprox(timeout=2000)
                logger.info(f"Num after iteration {iteration}: approx {count_approx}")
            
            iteration += 1
            
            # Checkpoint và cleanup mỗi 3 iterations
            if iteration % 2 == 0:
                if self.debug:
                    logger.info(f"Checkpointing and cleanup at iteration {iteration}")
                
                # Checkpoint để truncate lineage
                df = df.checkpoint(eager=True)
                
                # Unpersist tất cả df cũ
                for old_df, new_df in persisted_dfs:
                    if old_df is not df:  # Tránh unpersist df hiện tại
                        old_df.unpersist()
                    if new_df is not df:
                        new_df.unpersist()
                
                # Clear list
                persisted_dfs.clear()
         
        # Final cleanup trước khi tạo train/val split
        for old_df, new_df in persisted_dfs:
            if old_df is not df:
                old_df.unpersist()
            if new_df is not df:
                new_df.unpersist()
        persisted_dfs.clear()
        
        # Window để tạo row_id cho train/val split
        w = Window.orderBy(self.ts_col)
        df = df.withColumn("row_id", F.row_number().over(w))
        total = df.count()
        train_count = int(total * train_ratio)
        
        train_df = df.filter(F.col("row_id") <= train_count).drop("row_id")
        val_df = df.filter(F.col("row_id") > train_count).drop("row_id")
        
        train_users = train_df.select(self.user_col).distinct()
        train_items = train_df.select(self.item_col).distinct()
        
        val_df = val_df.join(train_users, on=self.user_col, how="inner") \
                       .join(train_items, on=self.item_col, how="inner")
        
        all_users = df.select(self.user_col).distinct()
        all_items = df.select(self.item_col).distinct()

        coldstart_users = all_users.subtract(train_users)
        coldstart_items = all_items.subtract(train_items)
        df.unpersist()
        
        return train_df, val_df, coldstart_users, coldstart_items