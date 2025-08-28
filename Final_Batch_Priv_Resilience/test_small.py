#!/usr/bin/env python3
"""
Teste pequeno para validar o código paralelo
"""
import Both_Correcto_Parallel as bcp

print("=== TESTE RÁPIDO ===")
print("Testando apenas modelo ER com 3 trials...")

try:
    df = bcp.run_trials(trials=3, model='ER', N=11, T=50)  # Muito reduzido para teste
    print("\n✅ SUCESSO!")
    print(f"Resultados:")
    print(df)
    print(f"\nFicheiro Excel criado: Both_ER_11x11_optimized.xlsx")
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
