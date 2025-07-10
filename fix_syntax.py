#!/usr/bin/env python3
"""
Script to fix syntax errors in app.py
"""

def fix_syntax_errors():
    """Fix syntax errors in app.py"""
    
    # Read the file
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix emoji characters
    content = content.replace('🏆', '')
    content = content.replace('🎯', '')
    content = content.replace('🚀', '')
    content = content.replace('📊', '')
    content = content.replace('💰', '')
    content = content.replace('📈', '')
    content = content.replace('⏰', '')
    content = content.replace('🗺️', '')
    content = content.replace('💡', '')
    content = content.replace('🎯', '')
    content = content.replace('⚠️', '')
    content = content.replace('✅', '')
    content = content.replace('📋', '')
    content = content.replace('🎉', '')
    content = content.replace('🔥', '')
    content = content.replace('🚗', '')
    content = content.replace('⚖️', '')
    content = content.replace('🌙', '')
    content = content.replace('📊', '')
    content = content.replace('⚡', '')
    
    # Fix R² to R2
    content = content.replace('R²', 'R2')
    
    # Fix f-string formatting issues in markdown strings
    # Replace problematic f-string formatting in markdown
    content = content.replace('**{best_mae:.2f} trips**', '**{best_mae:.2f} trips**')
    
    # Write the fixed content back
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Syntax errors fixed in app.py")

if __name__ == "__main__":
    fix_syntax_errors() 