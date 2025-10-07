# GitHub Setup Script for XAI Load Forecasting
# This script will help you push the project to GitHub

Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                ║" -ForegroundColor Cyan
Write-Host "║         XAI Load Forecasting - GitHub Setup                    ║" -ForegroundColor Cyan
Write-Host "║                                                                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Get GitHub username
$username = Read-Host "Enter your GitHub username"

# Get repository name (default: xai-load-forecasting)
$repoName = Read-Host "Enter repository name (press Enter for 'xai-load-forecasting')"
if ([string]::IsNullOrWhiteSpace($repoName)) {
    $repoName = "xai-load-forecasting"
}

Write-Host "`n📋 Repository Details:" -ForegroundColor Yellow
Write-Host "  Username: $username" -ForegroundColor White
Write-Host "  Repository: $repoName" -ForegroundColor White
Write-Host "  URL: https://github.com/$username/$repoName`n" -ForegroundColor White

# Confirm
$confirm = Read-Host "Continue? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Setup cancelled." -ForegroundColor Red
    exit
}

Write-Host "`n🚀 Setting up GitHub repository...`n" -ForegroundColor Green

# Add remote
Write-Host "Adding remote origin..." -ForegroundColor Cyan
git remote add origin "https://github.com/$username/$repoName.git"

# Rename branch to main
Write-Host "Renaming branch to main..." -ForegroundColor Cyan
git branch -M main

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
Write-Host "(You may be prompted for GitHub credentials)`n" -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                                                                ║" -ForegroundColor Green
    Write-Host "║         ✅ Successfully Pushed to GitHub! ✅                    ║" -ForegroundColor Green
    Write-Host "║                                                                ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green
    
    Write-Host "🎉 Your repository is now live at:" -ForegroundColor Yellow
    Write-Host "   https://github.com/$username/$repoName`n" -ForegroundColor Cyan
    
    Write-Host "📝 Next steps:" -ForegroundColor Yellow
    Write-Host "   1. Go to https://github.com/$username/$repoName/settings" -ForegroundColor White
    Write-Host "   2. Add repository description and topics" -ForegroundColor White
    Write-Host "   3. Enable GitHub Pages (optional)" -ForegroundColor White
    Write-Host "   4. Add repository topics: machine-learning, forecasting, explainable-ai, shap, lightgbm`n" -ForegroundColor White
} else {
    Write-Host "`n❌ Push failed. Please check:" -ForegroundColor Red
    Write-Host "   1. Repository exists on GitHub" -ForegroundColor White
    Write-Host "   2. You have correct permissions" -ForegroundColor White
    Write-Host "   3. GitHub credentials are correct`n" -ForegroundColor White
}
