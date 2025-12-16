
from app import app
import sys

def verify():
    print("Testing App Startup...")
    try:
        with app.test_client() as client:
            print("GET / ...")
            resp = client.get('/')
            if resp.status_code == 200:
                print("✅ GET / Success")
                content = resp.data.decode('utf-8')
                
                # Check for Scenario Simulator
                if "Simulator Skenario" in content and "Simulasikan" in content:
                     print("✅ Scenario Simulator UI found")
                else:
                     print("❌ Scenario Simulator section MISSING")
                     sys.exit(1)
                     
                # Check that strict Counterfactual Analysis is GONE
                # The label was "Analisis Skenario What-if (H+1)"
                # The new simulator label is "Simulator Skenario (H+1, H+3, H+7)"
                
                # So we check for the OLD specific string
                if "Analisis Skenario What-if (H+1)" in content:
                     print("❌ Old Auto-Counterfactual section STILL PRESENT")
                     sys.exit(1)
                else:
                    print("✅ Old Auto-Counterfactual section REMOVED")

            else:
                print(f"❌ GET / Failed with {resp.status_code}")
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ Verification crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
