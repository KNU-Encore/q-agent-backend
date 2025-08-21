import json
import os
import redis
from dotenv import load_dotenv

from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
from langchain_core.tools import Tool

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)


class SQLQueryAnalyzer:
    def __init__(self):
        self.llm = None
        self.vectorstore = None
        self.report_agent_executor = None
        self.rewrite_agent_executor = None
        self.initialize_components()

    def safe_numeric_value(self, value, default=0):
        if value is None:
            return default
        try:
            if isinstance(value, (int, float)):
                return value
            return float(value) if value != '' else default
        except (ValueError, TypeError):
            return default

    def safe_division(self, numerator, denominator, default=0):
        num = self.safe_numeric_value(numerator, 0)
        den = self.safe_numeric_value(denominator, 1)

        if den == 0:
            return default
        return num / den

    def format_metric_display(self, value, default_display="N/A"):
        if value is None:
            return default_display
        try:
            if isinstance(value, (int, float)):
                if value == int(value):
                    return f"{int(value):,}"
                else:
                    return f"{value:.3f}"
            return str(value)
        except:
            return default_display

    def initialize_components(self):
        try:
            load_dotenv()

            self.llm = ChatOpenAI(
                temperature=0.2,
                max_tokens=4096,
                model_name="gpt-4o-mini"
            )

            self.setup_knowledge_base()
            self.setup_agents()

            return True
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            return False

    def setup_knowledge_base(self):
        try:
            html_files = [
                "slow_query_log.html",
                "optimizing-slow-sql-queries.html",
                "order-by-optimization.html",
            ]

            docs = []
            for file in html_files:
                try:
                    if os.path.exists(file):
                        loader = BSHTMLLoader(file)
                        docs.extend(loader.load())
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")

            if docs:
                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                all_splits = text_splitter.split_documents(docs)
                self.vectorstore = FAISS.from_documents(all_splits, embedding=embeddings)

        except Exception as e:
            print(f"Knowledge base setup error: {str(e)}")

    def search_similar_examples(self, query: str) -> str:
        if not self.vectorstore:
            return "No knowledge base available."
        try:
            retrieved_docs = self.vectorstore.similarity_search(query, k=3)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            return f"Similar examples and best practices:\n\n{context}"
        except Exception as e:
            return f"Error searching examples: {str(e)}"

    def analyze_execution_plan_detailed(self, plan_tree: str) -> str:
        try:

            if not plan_tree or plan_tree.strip() == "":
                return "No execution plan available for analysis."

            analysis_prompt = f"""
            Analyze the following MySQL 8.0 execution plan in detail:

            Execution Plan:
            {plan_tree}

            Please provide comprehensive analysis including:

            1. **Operation Breakdown**:
               - Identify each operation in the plan tree
               - Explain the execution flow from inner to outer operations
               - Calculate total cost and time estimates

            2. **Performance Bottlenecks**:
               - Identify the most expensive operations
               - Find operations with high actual vs estimated times
               - Look for inefficient join strategies
               - Identify full table scans and missing indexes

            3. **Resource Usage Analysis**:
               - Memory usage (sort operations, joins)
               - CPU intensive operations
               - I/O patterns and disk access

            4. **Row Processing Efficiency**:
               - Compare rows examined vs rows returned
               - Identify operations with poor selectivity
               - Find unnecessary data processing

            5. **Join Analysis**:
               - Analyze join algorithms (nested loop, hash, etc.)
               - Evaluate join order efficiency
               - Check for proper index usage in joins

            6. **Sort Operations**:
               - Identify expensive sort operations
               - Check for filesort usage
               - Analyze sort buffer requirements

            Focus on specific numeric values and provide actionable insights.
            """

            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            return response.content
        except Exception as e:
            return f"Error analyzing execution plan: {str(e)}"

    def format_schema_info(self, schema_info):

        if not schema_info:
            return "### Database Schema: Not Available\n⚠️ Analysis will be limited without schema information."

        try:
            if isinstance(schema_info, dict):
                formatted_schema = "### Database Schema Information:\n"
                for key, value in schema_info.items():
                    formatted_schema += f"**{key}**: {value}\n"
                return formatted_schema
            elif isinstance(schema_info, str):
                return f"### Database Schema:\n{schema_info}"
            else:
                return f"### Database Schema:\n```json\n{json.dumps(schema_info, indent=2)}\n```"
        except Exception as e:
            return f"### Database Schema: Error formatting schema - {str(e)}"

    def setup_agents(self):
        try:
            tools = [
                Tool(
                    name="search_similar_examples",
                    description="Search for similar SQL performance examples and best practices from the knowledge base",
                    func=self.search_similar_examples
                ),
                Tool(
                    name="analyze_execution_plan",
                    description="Analyze a MySQL execution plan tree in detail",
                    func=self.analyze_execution_plan_detailed
                )
            ]

            report_system_message = """You are an expert SQL Performance Analysis Agent specializing in MySQL query optimization.

            Your role is to generate comprehensive performance analysis reports based on:
            - SQL query structure and complexity
            - MySQL execution plans (plan trees)
            - Performance metrics (execution time, rows examined/sent)
            - Database schema information (when available)

            ## Report Structure:

            ### 🔍 Query Analysis Overview
            - **Query Type & Complexity**: Detailed classification
            - **Tables & Relationships**: Schema-based analysis (utilize provided schema information)
            - **Operation Summary**: High-level execution overview
            - **Schema Context**: How schema affects query performance

            ### ⚡ Execution Plan Deep Dive
            - **Critical Path Analysis**: Most expensive operations with timing
            - **Join Strategy Evaluation**: Algorithm efficiency and order
            - **Index Utilization**: Current usage and gaps (reference schema indexes if available)
            - **Sort Operations**: Memory usage and filesort analysis
            - **Row Processing**: Efficiency metrics and bottlenecks

            ### 🚨 Performance Issues Identified
            For each issue provide:
            - **Issue Description**: Specific problem
            - **Performance Impact**: Quantified cost
            - **Root Cause**: Technical explanation (consider schema constraints)
            - **Severity**: Critical/High/Medium/Low

            ### 📊 Performance Metrics Analysis
            - **Execution Time Breakdown**: Where time is spent
            - **Resource Utilization**: CPU, Memory, I/O analysis
            - **Efficiency Ratios**: Rows examined vs returned
            - **Scalability Concerns**: Performance with data growth

            ### 🎯 Optimization Priorities
            1. **Critical Issues**: Immediate attention required
            2. **High Impact**: Significant performance gains
            3. **Quick Wins**: Low effort, good returns
            4. **Long-term**: Structural improvements (consider schema design)

            ### 📈 Expected Improvements
            - **Query Time Reduction**: Estimated improvements
            - **Resource Savings**: CPU, Memory, I/O reductions
            - **Scalability Enhancement**: Better performance curves

            **IMPORTANT**: Always reference and utilize the provided database schema information when available.
            Use schema details to provide more accurate analysis of table structures, relationships, and optimization opportunities.
            Handle cases where some performance metrics or schema information may be unavailable (N/A values).
            """

            rewrite_system_message = """You are an expert SQL Query Optimization Agent specializing in MySQL query rewrites.

            Your role is to generate optimized query versions based on:
            - Performance analysis findings
            - Execution plan bottlenecks
            - Database schema constraints and opportunities
            - MySQL-specific optimization techniques

            ## Rewrite Structure:

            ### 🔧 Primary Optimized Query
            ```sql
            -- Optimized version with detailed comments
            -- Consider schema-specific optimizations
            ```

            ### 🔄 Alternative Query Approaches
            - **Approach 1**: [Description and use case based on schema]
            ```sql
            -- Alternative implementation
            ```
            - **Approach 2**: [Description and use case]
            ```sql  
            -- Another alternative leveraging schema structure
            ```

            ### 📐 Index Recommendations
            ```sql
            -- Essential indexes for optimal performance
            -- Based on schema structure and query patterns
            CREATE INDEX idx_name ON table(columns);
            ```

            ### 🎛️ Optimization Techniques Applied
            1. **Join Optimization**: Order and algorithm improvements (schema-aware)
            2. **Index Utilization**: Better index usage patterns
            3. **Query Restructuring**: Logical improvements
            4. **Subquery Optimization**: Elimination or improvement
            5. **Sort Optimization**: Reducing sort overhead
            6. **Schema-Specific**: Optimizations based on table structure

            ### 📊 Performance Expectations
            - **Query Time**: Expected reduction percentage
            - **Rows Examined**: Reduction in data scanning
            - **Memory Usage**: Sort and join memory optimization
            - **CPU Usage**: Processing efficiency improvements

            ### ⚖️ Trade-off Analysis
            - **Benefits**: Performance gains and improvements
            - **Considerations**: Potential impacts or limitations
            - **Recommendations**: When to use each approach

            **IMPORTANT**: Always leverage the provided database schema information for accurate optimization.
            Use schema details to suggest proper indexes, table relationships, and MySQL-specific features.
            Provide executable, tested query patterns with MySQL-specific optimizations.
            Handle cases where some performance metrics or schema information may not be available.
            """

            report_prompt = ChatPromptTemplate.from_messages([
                ("system", report_system_message),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])

            report_agent = create_openai_functions_agent(self.llm, tools, report_prompt)
            self.report_agent_executor = AgentExecutor(
                agent=report_agent,
                tools=tools,
                verbose=True,
                max_iterations=6
            )

            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system", rewrite_system_message),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")]
            )

            rewrite_agent = create_openai_functions_agent(self.llm, tools, rewrite_prompt)
            self.rewrite_agent_executor = AgentExecutor(
                agent=rewrite_agent,
                tools=tools,
                verbose=True,
                max_iterations=6
            )

        except Exception as e:
            print(f"Agent setup error: {str(e)}")

    def generate_comprehensive_analysis(self, json_input: dict, schema_info: dict = None):
        try:
            metadata = json_input.get('query_metadata', {})
            explain_analyze = json_input.get('explain_analyze', {})

            query = metadata.get('query', 'No query available')
            plan_tree = explain_analyze.get('plan_tree', 'No execution plan available')

            db_schema = json_input.get('db_schema', {})
            if schema_info is None:
                schema_info = db_schema.get('schema_info', "No schema available")

            performance_metrics = {
                'qt': self.safe_numeric_value(metadata.get('qt')),
                'lt': self.safe_numeric_value(metadata.get('lt')),
                'rsent': self.safe_numeric_value(metadata.get('rsent')),
                'rexp': self.safe_numeric_value(metadata.get('rexp')),
                'timestamp': metadata.get('timestamp', 'Not specified')
            }

            efficiency = self.safe_division(performance_metrics['rsent'], performance_metrics['rexp']) * 100

            data_availability_note = ""
            missing_metrics = []

            if metadata.get('rsent') is None:
                missing_metrics.append("Rows Sent")
            if metadata.get('rexp') is None:
                missing_metrics.append("Rows Examined")
            if metadata.get('lt') is None:
                missing_metrics.append("Lock Time")

            if missing_metrics:
                data_availability_note = f"\n⚠️ **Note**: Some metrics are not available: {', '.join(missing_metrics)}. Analysis will focus on available data."

            formatted_schema = self.format_schema_info(schema_info)

            analysis_context = f"""
            ## SQL Performance Analysis Request
            {data_availability_note}

            ### Query to Analyze:
            ```sql
            {query}
            ```

            ### Performance Metrics:
            - **Execution Time**: {self.format_metric_display(performance_metrics['qt'])} seconds
            - **Lock Time**: {self.format_metric_display(performance_metrics['lt'])} seconds  
            - **Rows Returned**: {self.format_metric_display(performance_metrics['rsent'])}
            - **Rows Examined**: {self.format_metric_display(performance_metrics['rexp'])}
            - **Efficiency**: {efficiency:.2f}% (calculated from available data)
            - **Timestamp**: {performance_metrics['timestamp']}

            ### MySQL Execution Plan:
            ```
            {plan_tree}
            ```

            {formatted_schema}

            ### Analysis Requirements:
            1. **Schema-Aware Analysis**: Utilize the provided schema information for accurate table structure analysis
            2. Analyze the execution plan tree structure (if available)
            3. Identify performance bottlenecks from available metrics
            4. Search for similar optimization examples
            5. Provide detailed technical analysis with schema context
            6. Focus on MySQL-specific optimization opportunities
            7. Consider table relationships and constraints from schema
            8. Handle missing metric data gracefully

            **Key Focus Areas:**
            - Leverage schema information for index recommendations
            - Consider table relationships for join optimization
            - Use schema constraints for query rewrite opportunities
            - Execution time: {self.format_metric_display(performance_metrics['qt'])} seconds
            - Efficiency: {efficiency:.2f}% (based on available row count data)
            """

            print("🔍 Generating Performance Analysis Report...")
            report_result = self.report_agent_executor.invoke({"input": analysis_context})
            performance_report = report_result["output"]

            print("\n🔧 Generating Optimized Query Rewrites...")
            rewrite_context = f"""
            {analysis_context}

            ### Performance Analysis Summary:
            {performance_report[:1500] if performance_report else "Performance report not available"}...

            ### Rewrite Requirements:
            1. **Schema-Based Optimization**: Use schema information for accurate table structure optimization
            2. Focus on optimizing the query execution (current time: {self.format_metric_display(performance_metrics['qt'])} seconds)
            3. Improve efficiency where possible (current: {efficiency:.2f}%)
            4. Optimize subquery operations and UNION clauses
            5. Reduce table scan overhead using schema-aware indexing
            6. Provide multiple optimization approaches based on schema constraints
            7. Include necessary index recommendations using actual table structure
            8. Consider optimization even when some metrics are unavailable
            9. Leverage table relationships and foreign keys from schema

            **Schema Context for Optimization:**
            {formatted_schema}

            Generate optimized query versions with detailed explanations and schema-aware improvements.
            """

            rewrite_result = self.rewrite_agent_executor.invoke({"input": rewrite_context})
            query_rewrite = rewrite_result["output"]

            return {
                "performance_report": performance_report,
                "query_rewrite": query_rewrite,
                "metrics": performance_metrics,
                "efficiency": efficiency,
                "missing_metrics": missing_metrics,
                "data_quality": "partial" if missing_metrics else "complete",
                "schema_available": bool(schema_info and schema_info != "No schema available")
            }

        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                "error": f"Analysis failed: {str(e)}",
                "performance_report": None,
                "query_rewrite": None,
                "metrics": None,
                "efficiency": 0,
                "schema_available": False
            }


def run_ai_report_generation(session_id: str):
    input_key = f'analysis_inputs:{session_id}'
    report_key = f'report:{session_id}'

    try:
        input_data_json = redis_client.get(input_key)
        if not input_data_json:
            raise ValueError(f'Session id ({session_id}) not found in Redis.')

        analyzer = SQLQueryAnalyzer()
        result = analyzer.generate_comprehensive_analysis(json_input=json.loads(input_data_json))

        report = {
            'status': 'complete',
            'result': result,
        }
        redis_client.set(report_key, json.dumps(report), ex=3600)
        print(f'[{session_id}] Report saved to redis successfully')

    except Exception as e:
        print(f'[{session_id}] Background task failed: {e}')
        report = {
            'status': 'failed',
            'result': str(e),
        }
        redis_client.set(report_key, json.dumps(report), ex=3600)
