#!/usr/bin/env python3
"""
Agentic RAG v2 - Command Line Interface
Execute agentic RAG queries from the command line
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import config, load_config
from core.agent_orchestrator import AgentOrchestrator
from examples.example_queries import get_example_query, ALL_QUERIES


def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_result(result: dict, verbose: bool = False):
    """
    Print query result in formatted way
    
    Args:
        result: Result dictionary from agent
        verbose: Show detailed information
    """
    print_header("ANSWER")
    print(result['final_answer'])
    
    print(f"\n{'─'*80}")
    print("METADATA")
    print(f"{'─'*80}")
    print(f"  Iterations: {result['iterations']}/{result['config']['max_iterations']}")
    print(f"  Quality Score: {result['quality_score']:.3f}")
    print(f"  Contexts Used: {result['context_count']}")
    print(f"  Total Time: {result['total_time']}s")
    print(f"  Sources: {', '.join(result['sources'])}")
    
    if verbose and result.get('iteration_history'):
        print(f"\n{'─'*80}")
        print("ITERATION HISTORY")
        print(f"{'─'*80}")
        
        for iteration in result['iteration_history']:
            print(f"\n  Iteration {iteration['iteration']}:")
            print(f"    Query: {iteration['query'][:60]}...")
            print(f"    Retrieved: {iteration['tool_results']}")
            print(f"    Quality: {iteration['quality_score']:.3f}")
            print(f"    Time: {iteration['time']:.2f}s")
            
            reflection = iteration.get('reflection', {})
            if reflection:
                print(f"    Relevance: {reflection.get('relevance_score', 0):.3f}")
                print(f"    Coverage: {reflection.get('coverage_score', 0):.3f}")
                print(f"    Confidence: {reflection.get('confidence_score', 0):.3f}")
                
                gaps = reflection.get('gaps', [])
                if gaps:
                    print(f"    Gaps: {', '.join(gaps)}")


def single_query(query: str, args):
    """
    Execute single query
    
    Args:
        query: Query string
        args: Command line arguments
    """
    print_header(f"AGENTIC RAG v2 - Single Query")
    print(f"Query: {query}\n")
    
    # Initialize agent
    print("🔧 Initializing Agent...")
    agent = AgentOrchestrator(
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
        enable_web_search=args.enable_web_search,
        verbose=args.verbose
    )
    
    # Execute query
    result = agent.query(query)
    
    # Print result
    print_result(result, verbose=args.verbose)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 Results saved to: {output_path}")
    
    # Cleanup
    agent.close()


def interactive_mode(args):
    """
    Interactive query mode
    
    Args:
        args: Command line arguments
    """
    print_header("AGENTIC RAG v2 - Interactive Mode")
    print("Type your questions below. Type 'exit' or 'quit' to exit.")
    print("Type 'example' to see example queries.")
    print("Type 'config' to see current configuration.\n")
    
    # Initialize agent once
    agent = AgentOrchestrator(
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
        enable_web_search=args.enable_web_search,
        verbose=args.verbose
    )
    
    query_count = 0
    
    try:
        while True:
            # Get user input
            try:
                query = input(f"\n🔍 Query [{query_count + 1}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break
            
            # Handle special commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'example':
                print("\n📝 Example Queries:")
                for i, ex in enumerate(ALL_QUERIES[:10], 1):
                    print(f"  {i}. {ex}")
                continue
            
            if query.lower() == 'config':
                print("\n⚙️  Current Configuration:")
                print(f"  Max Iterations: {args.max_iterations}")
                print(f"  Quality Threshold: {args.quality_threshold}")
                print(f"  Web Search: {'Enabled' if args.enable_web_search else 'Disabled'}")
                print(f"  Verbose: {'Yes' if args.verbose else 'No'}")
                continue
            
            if not query:
                print("  ⚠️  Please enter a query.")
                continue
            
            # Execute query
            query_count += 1
            print(f"\n{'─'*80}")
            
            try:
                result = agent.query(query)
                print_result(result, verbose=args.verbose)
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
    
    finally:
        # Cleanup
        agent.close()


def batch_evaluation(args):
    """
    Batch evaluation mode
    
    Args:
        args: Command line arguments
    """
    print_header("AGENTIC RAG v2 - Batch Evaluation")
    
    # Load queries
    queries = ALL_QUERIES[:args.sample_size] if args.sample_size > 0 else ALL_QUERIES
    
    print(f"📝 Evaluating {len(queries)} queries...")
    print(f"⚙️  Config: max_iter={args.max_iterations}, threshold={args.quality_threshold}\n")
    
    # Initialize agent
    agent = AgentOrchestrator(
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
        enable_web_search=args.enable_web_search,
        verbose=False  # Disable verbose for batch
    )
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query[:60]}... ", end='', flush=True)
        
        try:
            result = agent.query(query)
            results.append(result)
            print(f"✅ (iter={result['iterations']}, quality={result['quality_score']:.2f}, time={result['total_time']:.1f}s)")
        except Exception as e:
            print(f"❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Print summary
    if results:
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        total_queries = len(results)
        avg_iterations = sum(r['iterations'] for r in results) / total_queries
        avg_quality = sum(r['quality_score'] for r in results) / total_queries
        avg_time = sum(r['total_time'] for r in results) / total_queries
        
        print(f"  Total Queries: {total_queries}")
        print(f"  Success Rate: {total_queries}/{len(queries)} ({total_queries/len(queries)*100:.1f}%)")
        print(f"  Avg Iterations: {avg_iterations:.2f}")
        print(f"  Avg Quality Score: {avg_quality:.3f}")
        print(f"  Avg Time: {avg_time:.2f}s")
        
        # Iteration distribution
        iter_dist = {}
        for r in results:
            iter_count = r['iterations']
            iter_dist[iter_count] = iter_dist.get(iter_count, 0) + 1
        
        print(f"\n  Iteration Distribution:")
        for iter_num in sorted(iter_dist.keys()):
            count = iter_dist[iter_num]
            percent = count / total_queries * 100
            print(f"    {iter_num} iterations: {count} queries ({percent:.1f}%)")
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': {
                        'total_queries': total_queries,
                        'avg_iterations': avg_iterations,
                        'avg_quality': avg_quality,
                        'avg_time': avg_time,
                        'iteration_distribution': iter_dist
                    },
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\n📝 Results saved to: {output_path}")
    
    # Cleanup
    agent.close()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Agentic RAG v2 - Intelligent Question Answering System",
        epilog="""
Examples:
  # Single query
  python run_agentic.py "What is perovskite solar cell efficiency?"
  
  # Interactive mode
  python run_agentic.py --interactive
  
  # Batch evaluation
  python run_agentic.py --evaluate --sample-size 10
  
  # With custom parameters
  python run_agentic.py "Your query" --max-iterations 5 --quality-threshold 0.8 --verbose
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Positional argument
    parser.add_argument(
        'query',
        nargs='?',
        help='Query string for single query mode'
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive query mode'
    )
    mode_group.add_argument(
        '--evaluate', '-e',
        action='store_true',
        help='Batch evaluation mode'
    )
    
    # Agent configuration
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=config.max_iterations if config else 3,
        help='Maximum number of iterations (default: 3)'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=config.quality_threshold if config else 0.7,
        help='Quality threshold for stopping (default: 0.7)'
    )
    parser.add_argument(
        '--enable-web-search',
        action='store_true',
        help='Enable web search tool'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Evaluation options
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10,
        help='Number of queries for evaluation (default: 10, 0 for all)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        load_config()
    except Exception as e:
        print(f"⚠️  Configuration warning: {e}")
        print("   Please ensure .env file is configured correctly.")
        return 1
    
    # Route to appropriate mode
    try:
        if args.interactive:
            interactive_mode(args)
        elif args.evaluate:
            batch_evaluation(args)
        elif args.query:
            single_query(args.query, args)
        else:
            parser.print_help()
            print("\n💡 Tip: Use --interactive for interactive mode")
            print("        Use --evaluate for batch evaluation")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

