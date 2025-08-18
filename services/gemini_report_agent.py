import json
import os
import re
from typing import Dict, Any, List, Optional

import redis
from dotenv import load_dotenv
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)


class SimpleHallucinationDetector:
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.1):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    def detect_hallucinations(self, report: str, original_data: dict) -> dict:
        try:
            context_data = f"""
            Original Query Data:
            Query: {original_data.get('query', 'N/A')}
            Execution Time: {original_data.get('performance_metrics', {}).get('qt', 'N/A')} seconds
            Lock Time: {original_data.get('performance_metrics', {}).get('lt', 'N/A')} seconds
            Rows Sent: {original_data.get('performance_metrics', {}).get('rsent', 'N/A')}
            Rows Examined: {original_data.get('performance_metrics', {}).get('rexp', 'N/A')}
            Execution Plan: {str(original_data.get('plan_tree', 'N/A'))[:1000]}
            Schema Info: {str(original_data.get('schema_info', 'N/A'))[:500]}
            """

            verification_prompt = f"""
            You are a SQL performance analysis fact-checker. Your task is to verify the accuracy of a SQL performance analysis report against the original query data.

            ORIGINAL DATA:
            {context_data}

            ANALYSIS REPORT TO VERIFY:
            {report}


            Please carefully review the analysis report and verify its accuracy against the original data. Focus on:

            "Here is a statement: {report} Is the above statement correct?",

            "Does the statement '{report}' accurately reflect the truth?",

            "Please verify if this statement is accurate: {report}",

            "Is the following claim factually correct: '{report}'?",

            "Evaluate the truthfulness of: {report}",

            "Can you confirm the accuracy of this statement: '{report}'?",

            "How accurate is this claim: {report}?",

            "Verify the factual correctness of: '{report}'",

            1. **Factual Claims Verification:**
               - Are the performance metrics (execution time, rows sent/examined) correctly stated?
               - Are the technical statements about SQL operations accurate?
               - Do the optimization recommendations align with the actual query structure?
               - Are any numerical values or calculations correct?

            2. **Technical Accuracy Assessment:**
               - Are execution plan interpretations consistent with the provided plan?
               - Are the identified bottlenecks supported by the actual data?
               - Are index recommendations appropriate for the query structure?
               - Are join algorithm assessments accurate?

            3. **Consistency Check:**
               - Does the analysis stay within the bounds of the provided data?
               - Are conclusions logically derived from the available information?
               - Are there any unsupported claims or speculative statements?

            Please provide your assessment in the following format:

            **RELIABILITY SCORE:** [Score from 1-10, where 10 is completely accurate]

            **CONFIDENCE LEVEL:** [High/Medium/Low]

            **VERIFIED FACTS:** [Number of facts that were verified as correct]

            **QUESTIONABLE CLAIMS:** [Number of claims that seem incorrect or unsupported]

            **DETAILED ASSESSMENT:**
            [Provide a detailed explanation of your findings, including:]
            - What claims were verified as accurate
            - What claims appear questionable or incorrect
            - Any unsupported statements or potential hallucinations
            - Overall assessment of the report's reliability

            **SPECIFIC ISSUES FOUND:**
            [List any specific factual errors, inconsistencies, or unsupported claims you identified]
            """

            response = self.llm.invoke([HumanMessage(content=verification_prompt)])

            content = response.content

            reliability_score = 5
            score_match = re.search(
                r"RELIABILITY SCORE.*?(\d+(?:\.\d+)?)", content, re.IGNORECASE
            )
            if score_match:
                try:
                    reliability_score = float(score_match.group(1))
                except ValueError:
                    reliability_score = 5

            confidence_level = "Medium"
            confidence_match = re.search(
                r"CONFIDENCE LEVEL.*?(High|Medium|Low)", content, re.IGNORECASE
            )
            if confidence_match:
                confidence_level = confidence_match.group(1).title()

            verified_facts = 0
            verified_match = re.search(
                r"VERIFIED FACTS.*?(\d+)", content, re.IGNORECASE
            )
            if verified_match:
                try:
                    verified_facts = int(verified_match.group(1))
                except ValueError:
                    verified_facts = 0

            questionable_claims = 0
            questionable_match = re.search(
                r"QUESTIONABLE CLAIMS.*?(\d+)", content, re.IGNORECASE
            )
            if questionable_match:
                try:
                    questionable_claims = int(questionable_match.group(1))
                except ValueError:
                    questionable_claims = 0

            assessment_match = re.search(
                r"DETAILED ASSESSMENT:\s*\n(.*?)(?=\*\*SPECIFIC ISSUES|$)",
                content,
                re.IGNORECASE | re.DOTALL,
            )
            detailed_analysis = (
                assessment_match.group(1).strip()
                if assessment_match
                else "No detailed assessment provided"
            )

            issues_match = re.search(
                r"SPECIFIC ISSUES FOUND:\s*\n(.*?)$", content, re.IGNORECASE | re.DOTALL
            )
            questionable_statements = (
                issues_match.group(1).strip()
                if issues_match
                else "No specific issues identified"
            )

            if (
                questionable_claims > 0
                and questionable_statements == "No specific issues identified"
            ):
                questionable_statements = f"Analysis detected {questionable_claims} potentially questionable claims. Review recommended."

            return {
                "reliability_score": round(reliability_score, 1),
                "confidence_level": confidence_level,
                "total_claims": verified_facts + questionable_claims,
                "verified_claims": verified_facts,
                "questionable_claims": questionable_claims,
                "detailed_analysis": detailed_analysis,
                "questionable_statements": questionable_statements,
                "methodology": "Single comprehensive prompt-based verification",
                "full_response": content,
            }

        except Exception as e:
            print(f"Error in hallucination detection: {str(e)}")
            return {
                "reliability_score": 0,
                "confidence_level": "Unknown",
                "total_claims": 0,
                "verified_claims": 0,
                "questionable_claims": 0,
                "detailed_analysis": f"Error occurred during verification: {str(e)}",
                "questionable_statements": "Verification failed due to error",
                "methodology": "Single prompt verification (failed)",
                "error": str(e),
            }


class AnalysisState(TypedDict):
    query: str
    plan_tree: str
    performance_metrics: Dict[str, Any]
    schema_info: Dict[str, Any]
    efficiency: float
    missing_metrics: List[str]
    data_availability_note: str
    formatted_schema: str
    analysis_context: str
    performance_report: Optional[str]
    query_rewrite: Optional[str]
    error: Optional[str]
    data_quality: str
    schema_available: bool
    hallucination_report: Optional[Dict[str, Any]]
    verified_report: Optional[str]
    fact_check_results: Optional[Dict[str, Any]]


class SQLQueryAnalyzer:
    def __init__(self):
        self.llm = None
        self.hallucination_detector = None
        self.vectorstore = None
        self.report_agent_executor = None
        self.rewrite_agent_executor = None
        self.db_schema = None
        self.graph = None
        self.initialize_components()

    def safe_numeric_value(self, value, default=0):
        if value is None:
            return default
        try:
            if isinstance(value, (int, float)):
                return value
            return float(value) if value != "" else default
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

            os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "my-default-user-agent")
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

            self.hallucination_detector = SimpleHallucinationDetector()

            self.setup_agents()
            self.setup_graph()

            return True
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            return False

    def load_query_data(self, json_file_path: str) -> dict:
        try:
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None

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
                    name="analyze_execution_plan",
                    description="Analyze a MySQL execution plan tree in detail",
                    func=self.analyze_execution_plan_detailed,
                )
            ]

            report_system_message = """You are an expert SQL Performance Analysis Agent specializing in MySQL query optimization.

            Your role is to generate comprehensive performance analysis reports based on:
            - SQL query structure and complexity
            - MySQL execution plans (plan trees)
            - Performance metrics (execution time, rows examined/sent)
            - Database schema information (when available)

            ## Report Structure (Use proper Markdown formatting):

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

            **IMPORTANT**: 
            - Always use proper Markdown formatting (headers, bold, code blocks, tables, lists)
            - Reference and utilize the provided database schema information when available
            - Use schema details to provide more accurate analysis of table structures, relationships, and optimization opportunities
            - Handle cases where some performance metrics or schema information may be unavailable (N/A values)
            - Format code snippets with proper SQL syntax highlighting using ```sql blocks
            - BE FACTUAL and stick to the actual data provided - avoid making unsupported claims
            """

            rewrite_system_message = """You are an expert SQL Query Optimization Agent specializing in MySQL query rewrites.

            Your role is to generate optimized query versions based on:
            - Performance analysis findings
            - Execution plan bottlenecks
            - Database schema constraints and opportunities
            - MySQL-specific optimization techniques

            ## Rewrite Structure (Use proper Markdown formatting):

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

            **IMPORTANT**: 
            - Always use proper Markdown formatting (headers, bold, code blocks, tables, lists)
            - Leverage the provided database schema information for accurate optimization
            - Use schema details to suggest proper indexes, table relationships, and MySQL-specific features
            - Provide executable, tested query patterns with MySQL-specific optimizations
            - Handle cases where some performance metrics or schema information may not be available
            - Format all SQL code with proper syntax highlighting using ```sql blocks
            - BE FACTUAL and base recommendations on actual query structure and provided data
            """

            report_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", report_system_message),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

            report_agent = create_openai_functions_agent(self.llm, tools, report_prompt)
            self.report_agent_executor = AgentExecutor(
                agent=report_agent, tools=tools, verbose=True, max_iterations=6
            )

            rewrite_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", rewrite_system_message),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

            rewrite_agent = create_openai_functions_agent(
                self.llm, tools, rewrite_prompt
            )
            self.rewrite_agent_executor = AgentExecutor(
                agent=rewrite_agent, tools=tools, verbose=True, max_iterations=6
            )

        except Exception as e:
            print(f"Agent setup error: {str(e)}")

    def setup_graph(self):

        def prepare_analysis(state: AnalysisState) -> AnalysisState:
            try:

                efficiency = (
                    self.safe_division(
                        state["performance_metrics"]["rsent"],
                        state["performance_metrics"]["rexp"],
                    )
                    * 100
                )

                missing_metrics = []
                if state["performance_metrics"].get("rsent") is None:
                    missing_metrics.append("Rows Sent")
                if state["performance_metrics"].get("rexp") is None:
                    missing_metrics.append("Rows Examined")
                if state["performance_metrics"].get("lt") is None:
                    missing_metrics.append("Lock Time")

                data_availability_note = ""
                if missing_metrics:
                    data_availability_note = f"\n⚠️ **Note**: Some metrics are not available: {', '.join(missing_metrics)}. Analysis will focus on available data."

                formatted_schema = self.format_schema_info(state["schema_info"])

                analysis_context = f"""
                ## SQL Performance Analysis Request
                {data_availability_note}

                ### Query to Analyze:
                ```sql
                {state['query']}
                ```

                ### Performance Metrics:
                - **Execution Time**: {self.format_metric_display(state['performance_metrics']['qt'])} seconds
                - **Lock Time**: {self.format_metric_display(state['performance_metrics']['lt'])} seconds  
                - **Rows Returned**: {self.format_metric_display(state['performance_metrics']['rsent'])}
                - **Rows Examined**: {self.format_metric_display(state['performance_metrics']['rexp'])}
                - **Efficiency**: {efficiency:.2f}% (calculated from available data)
                - **Timestamp**: {state['performance_metrics']['timestamp']}

                ### MySQL Execution Plan:
                ```
                {state['plan_tree']}
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
                9. **Use proper Markdown formatting throughout the analysis**

                **Key Focus Areas:**
                - Leverage schema information for index recommendations
                - Consider table relationships for join optimization
                - Use schema constraints for query rewrite opportunities
                - Execution time: {self.format_metric_display(state['performance_metrics']['qt'])} seconds
                - Efficiency: {efficiency:.2f}% (based on available row count data)
                """

                return {
                    **state,
                    "efficiency": efficiency,
                    "missing_metrics": missing_metrics,
                    "data_availability_note": data_availability_note,
                    "formatted_schema": formatted_schema,
                    "analysis_context": analysis_context,
                }

            except Exception as e:
                print(f"Error in prepare_analysis: {str(e)}")
                return {**state, "error": f"Preparation failed: {str(e)}"}

        def generate_report(state: AnalysisState) -> AnalysisState:
            try:
                report_result = self.report_agent_executor.invoke(
                    {"input": state["analysis_context"]}
                )

                performance_report = report_result["output"]

                return {**state, "performance_report": performance_report}

            except Exception as e:
                print(f"Error in generate_report: {str(e)}")
                return {**state, "error": f"Report generation failed: {str(e)}"}

        def generate_rewrite(state: AnalysisState) -> AnalysisState:
            try:

                rewrite_context = f"""
                {state['analysis_context']}

                ### Performance Analysis Summary:
                {state.get('performance_report', 'Performance report not available')[:1500] if state.get('performance_report') else "Performance report not available"}...

                ### Rewrite Requirements:
                1. **Schema-Based Optimization**: Use schema information for accurate table structure optimization
                2. Focus on optimizing the query execution (current time: {self.format_metric_display(state['performance_metrics']['qt'])} seconds)
                3. Improve efficiency where possible (current: {state['efficiency']:.2f}%)
                4. Optimize subquery operations and UNION clauses
                5. Reduce table scan overhead using schema-aware indexing
                6. Provide multiple optimization approaches based on schema constraints
                7. Include necessary index recommendations using actual table structure
                8. Consider optimization even when some metrics are unavailable
                9. Leverage table relationships and foreign keys from schema
                10. **Use proper Markdown formatting with SQL code blocks**

                **Schema Context for Optimization:**
                {state['formatted_schema']}

                Generate optimized query versions with detailed explanations and schema-aware improvements.
                """

                rewrite_result = self.rewrite_agent_executor.invoke(
                    {"input": rewrite_context}
                )

                query_rewrite = rewrite_result["output"]

                return {**state, "query_rewrite": query_rewrite}

            except Exception as e:
                print(f"Error in generate_rewrite: {str(e)}")
                return {**state, "error": f"Rewrite generation failed: {str(e)}"}

        def detect_hallucinations_node(state: AnalysisState) -> AnalysisState:
            try:
                original_data = {
                    "query": state["query"],
                    "plan_tree": state["plan_tree"],
                    "performance_metrics": state["performance_metrics"],
                    "schema_info": state["schema_info"],
                }

                combined_report = ""
                if state.get("performance_report"):
                    combined_report += (
                        f"PERFORMANCE ANALYSIS:\n{state['performance_report']}\n\n"
                    )
                if state.get("query_rewrite"):
                    combined_report += f"QUERY OPTIMIZATION:\n{state['query_rewrite']}"

                if not combined_report.strip():
                    return {
                        **state,
                        "error": "No report content available for hallucination detection",
                    }

                hallucination_report = (
                    self.hallucination_detector.detect_hallucinations(
                        combined_report, original_data
                    )
                )

                # Create verified report if reliability is good
                verified_report = None
                if hallucination_report.get("reliability_score", 0) >= 6:
                    verified_report = state.get("performance_report", "")
                    if hallucination_report.get(
                        "questionable_statements"
                    ) and "No specific issues" not in hallucination_report.get(
                        "questionable_statements", ""
                    ):
                        verified_report += f"\n\n### ⚠️ Verification Notes\n{hallucination_report.get('questionable_statements', '')}"

                return {
                    **state,
                    "hallucination_report": hallucination_report,
                    "verified_report": verified_report,
                    "fact_check_results": {
                        "reliability_score": hallucination_report.get(
                            "reliability_score", 0
                        ),
                        "total_claims": hallucination_report.get("total_claims", 0),
                        "verified_claims": hallucination_report.get(
                            "verified_claims", 0
                        ),
                    },
                }

            except Exception as e:
                print(f"Error in hallucination detection: {str(e)}")
                return {
                    **state,
                    "hallucination_report": {
                        "error": str(e),
                        "reliability_score": 0,
                        "confidence_level": "Unknown",
                    },
                }

        def finalize_results(state: AnalysisState) -> AnalysisState:
            try:
                data_quality = "partial" if state["missing_metrics"] else "complete"
                schema_available = bool(
                    state["schema_info"]
                    and state["schema_info"] != "No schema available"
                )

                return {
                    **state,
                    "data_quality": data_quality,
                    "schema_available": schema_available,
                }

            except Exception as e:
                print(f"Error in finalize_results: {str(e)}")
                return {**state, "error": f"Result finalization failed: {str(e)}"}

        workflow = StateGraph(AnalysisState)

        workflow.add_node("prepare", prepare_analysis)
        workflow.add_node("report", generate_report)
        workflow.add_node("rewrite", generate_rewrite)
        workflow.add_node("hallucination_check", detect_hallucinations_node)
        workflow.add_node("finalize", finalize_results)

        workflow.set_entry_point("prepare")
        workflow.add_edge("prepare", "report")
        workflow.add_edge("report", "rewrite")
        workflow.add_edge("rewrite", "hallucination_check")
        workflow.add_edge("hallucination_check", "finalize")
        workflow.add_edge("finalize", END)

        self.graph = workflow.compile()

    def generate_comprehensive_analysis(
        self, json_input: dict, schema_info: dict = None, save_markdown: bool = True
    ):
        try:
            metadata = json_input.get("query_metadata", {})
            explain_analyze = json_input.get("explain_analyze", {})

            query = metadata.get("query", "No query available")
            plan_tree = explain_analyze.get("plan_tree", "No execution plan available")

            db_schema = json_input.get("db_schema", {})
            if schema_info is None:
                schema_info = db_schema.get("schema_info", "No schema available")

            performance_metrics = {
                "qt": self.safe_numeric_value(metadata.get("qt")),
                "lt": self.safe_numeric_value(metadata.get("lt")),
                "rsent": self.safe_numeric_value(metadata.get("rsent")),
                "rexp": self.safe_numeric_value(metadata.get("rexp")),
                "timestamp": metadata.get("timestamp", "Not specified"),
            }

            initial_state = AnalysisState(
                query=query,
                plan_tree=plan_tree,
                performance_metrics=performance_metrics,
                schema_info=schema_info,
                efficiency=0.0,
                missing_metrics=[],
                data_availability_note="",
                formatted_schema="",
                analysis_context="",
                performance_report=None,
                query_rewrite=None,
                error=None,
                data_quality="unknown",
                schema_available=False,
                hallucination_report=None,
                verified_report=None,
                fact_check_results=None,
            )

            # print("🚀 Starting SQL query analysis with simple hallucination detection...")
            final_state = self.graph.invoke(initial_state)

            results = {
                "performance_report": final_state.get("performance_report"),
                "verified_report": final_state.get("verified_report"),
                "query_rewrite": final_state.get("query_rewrite"),
                "metrics": final_state.get("performance_metrics"),
                "efficiency": final_state.get("efficiency"),
                "missing_metrics": final_state.get("missing_metrics"),
                "data_quality": final_state.get("data_quality"),
                "schema_available": final_state.get("schema_available"),
                "hallucination_report": final_state.get("hallucination_report"),
                "fact_check_results": final_state.get("fact_check_results"),
                "error": final_state.get("error"),
            }

            return results

        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            import traceback

            traceback.print_exc()

            return {
                "error": f"Analysis failed: {str(e)}",
                "performance_report": None,
                "verified_report": None,
                "query_rewrite": None,
                "metrics": None,
                "efficiency": 0,
                "schema_available": False,
                "hallucination_report": None,
                "markdown_file": None,
            }


def run_gemini_report_generation(session_id: str):
    input_key = f"analysis_inputs:{session_id}"
    report_key = f"report:{session_id}"

    try:
        input_data_json = redis_client.get(input_key)
        if not input_data_json:
            raise ValueError(f"Session id ({session_id}) not found in Redis.")

        analyzer = SQLQueryAnalyzer()
        result = analyzer.generate_comprehensive_analysis(
            json_input=json.loads(input_data_json)
        )

        report = {
            "status": "complete",
            "result": result,
        }
        redis_client.set(report_key, json.dumps(report), ex=3600)
        print(f"[{session_id}] Report saved to redis successfully")

    except Exception as e:
        print(f"[{session_id}] Background task failed: {e}")
        report = {
            "status": "failed",
            "result": str(e),
        }
        redis_client.set(report_key, json.dumps(report), ex=3600)
