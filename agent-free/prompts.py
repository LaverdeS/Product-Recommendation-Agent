SYS_PROMPT = """You are a retail product expert.
                Carefully analyze the following conversation history
                and the current user query.
                Refer to the history and rephrase the current user query
                into a standalone query which can be used without the history
                for making search queries.
                Rephrase only if needed.
                Just return the query and do not answer it.
            """


FILTER_PROMPT = """Given the following schema of a dataframe table,
            your task is to figure out the best pandas query to
            filter the dataframe based on the user query which
            will be in natural language.

            The schema is as follows:

            #   Column        Non-Null Count  Dtype
            ---  ------        --------------  -----
            0   Product_ID    30 non-null     object
            1   Product_Name  30 non-null     object
            2   Category      30 non-null     object
            3   Price_USD     30 non-null     int64
            4   Rating        30 non-null     float64
            5   Description   30 non-null     object

            Category has values: ['Laptop', 'Tablet', 'Smartphone',
                                  'Smartwatch', 'Camera',
                                  'Headphones', 'Mouse', 'Keyboard',
                                  'Monitor', 'Charger']

            Rating ranges from 1 - 5 in floats

            You will try to figure out the pandas query focusing
            only on Category, Price_USD and Rating if the user mentions
            anything about these in their natural language query.
            Do not make up column names, only use the above.
            If not the pandas query should just return the full dataframe.
            Remember the dataframe name is df.

            Just return only the pandas query and nothing else.
            Do not return the results as markdown, just return the query

            User Query: {user_query}
            Pandas Query:
        """

RECOMMEND_PROMPT = """Act as an expert retail product advisor
                          Given the following table of products,
                          focus on the product attributes and description in the table
                          and based on the user query below do the following

                          - Recommend the most appropriate products based on the query
                          - Recommedation should have product name, price,  rating, description
                          - Also add a brief on why you recommend the product
                          - Do not make up products or recommend products not in the table
                          - If some specifications do not match focus on the ones which match and recommend
                          - If nothing matches recommend 5 random products from the table
                          - Do not generate anything else except the fields mentioned above

                        In case the user query is just a generic query or greeting
                        respond to them appropriately without recommending any products

                        Product Table:
                        {product_table}

                        User Query:
                        {user_query}

                        Recommendation:
                        """

USER_QUERIES = [
    "looking for a tablet with > 10 inch display and at least 64GB storage",
    "looking for a tablet",
]